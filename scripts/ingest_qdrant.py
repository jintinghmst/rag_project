"""
Stage 4: embed data/chunks/<key>.jsonl and load it into Qdrant.

BGE-M3 produces a dense vector and a learned sparse (lexical) vector in one
forward pass. Both are stored as named vectors on the same point, so retrieval
can fuse semantic similarity with exact term matching -- which is what recovers
queries that hinge on notation (Z0, S21, "equation 4.48") that dense vectors
alone handle poorly.

The ingest is incremental per document. A companion `<collection>__manifest`
collection records what was loaded for each book and from which chunk file, so a
build after adding one PDF embeds that book only -- and does so correctly against
a shared server, where the local data/ directory is no guide to what the index
already holds. A changed book has its old points deleted before the new ones go
in, so a book that shrinks does not leave orphans behind.

    python scripts/ingest_qdrant.py               # only what changed
    python scripts/ingest_qdrant.py --only hall_asi --force
    python scripts/ingest_qdrant.py --recreate    # wipe and rebuild everything
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
from search_qdrant import get_client  # noqa: E402

DENSE_DIM = 1024
MAX_LENGTH = 1024
PAYLOAD_INDEXES = {"book": "keyword", "chapter": "integer", "section": "keyword"}


def manifest_name(collection):
    return f"{collection}__manifest"


def read_manifest(client, collection):
    """{key: signature} for what is currently in the index."""
    name = manifest_name(collection)
    if not client.collection_exists(name):
        return {}
    points, _ = client.scroll(collection_name=name, limit=1000, with_payload=True)
    return {p.payload["key"]: p.payload.get("sig") for p in points}


def write_manifest(client, collection, key, doc, sig, n):
    from qdrant_client import models

    name = manifest_name(collection)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            # the manifest is a key/value table; Qdrant still wants a vector, so
            # it gets the smallest one that is legal
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
    client.upsert(
        collection_name=name,
        points=[models.PointStruct(
            id=doc["slot"],
            vector=[0.0],
            payload={"key": key, "title": doc["title"], "sig": sig, "chunks": n},
        )],
    )


def ensure_collection(client, collection):
    from qdrant_client import models

    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    for field, kind in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD if kind == "keyword"
            else models.PayloadSchemaType.INTEGER,
        )


def load_chunks(path, limit=None):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def signature(path, doc):
    h = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    return f"{h}:{doc['slot']}"


def embed_document(client, collection, model, key, rows, batch, quiet=False):
    from qdrant_client import models

    total = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        out = model.encode(
            [r["embed_text"] for r in chunk],
            batch_size=len(chunk),
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        points = []
        for row, dense, lex in zip(chunk, out["dense_vecs"], out["lexical_weights"]):
            payload = {k: v for k, v in row.items() if k != "embed_text"}
            points.append(models.PointStruct(
                id=row["id"],
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=[int(i) for i in lex.keys()],
                        values=[float(v) for v in lex.values()],
                    ),
                },
                payload=payload,
            ))
        client.upsert(collection_name=collection, points=points)
        total += len(points)
        if not quiet and (start % (batch * 25) == 0 or total == len(rows)):
            print(f"    {key}: {total}/{len(rows)}", flush=True)
    return total


def drop_document(client, collection, key):
    from qdrant_client import models

    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="book", match=models.MatchValue(value=key))]
        )),
    )


def run(keys=None, force=False, recreate=False, batch=8, limit=None, quiet=False):
    cfg = corpus.load()
    collection = os.environ.get("RAG_COLLECTION", cfg["collection"])
    client = get_client()
    model = None
    loaded, skipped = {}, []

    try:
        if recreate:
            for name in (collection, manifest_name(collection)):
                if client.collection_exists(name):
                    client.delete_collection(name)
        ensure_collection(client, collection)
        have = read_manifest(client, collection)

        for key, doc in cfg["documents"].items():
            if keys and key not in keys:
                continue
            path = corpus.doc_paths(key, doc)["chunks"]
            if not path.exists():
                continue
            sig = signature(path, doc)
            if not force and not recreate and have.get(key) == sig:
                skipped.append(key)
                continue

            rows = load_chunks(path, limit)
            if model is None:
                if not quiet:
                    print("  loading the embedding model (first run downloads ~2.3 GB)",
                          flush=True)
                from FlagEmbedding import BGEM3FlagModel
                model = BGEM3FlagModel(cfg["embed_model"], use_fp16=True)

            if key in have:
                drop_document(client, collection, key)
            n = embed_document(client, collection, model, key, rows, batch, quiet)
            write_manifest(client, collection, key, doc, sig, n)
            loaded[key] = n

        info = client.get_collection(collection)
        return collection, loaded, skipped, info.points_count
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="document keys to ingest")
    ap.add_argument("--limit", type=int, default=None, help="first N chunks per book (smoke test)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    ap.add_argument("--recreate", action="store_true", help="delete the collection first")
    args = ap.parse_args()

    collection, loaded, skipped, points = run(
        args.only, args.force, args.recreate, args.batch, args.limit)
    if skipped:
        print(f"unchanged: {', '.join(skipped)}")
    for key, n in loaded.items():
        print(f"loaded {key}: {n} points")
    where = os.environ.get("QDRANT_URL") or corpus.QDRANT_PATH
    print(f"\ncollection '{collection}': {points} points at {where}")


if __name__ == "__main__":
    main()
