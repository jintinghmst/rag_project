"""
Embed data/chunks.jsonl with BAAI/bge-m3 and load it into a local Qdrant store.

BGE-M3 produces a dense vector and a learned sparse (lexical) vector in one
forward pass. Both are stored as named vectors on the same point, so retrieval
can fuse semantic similarity with exact term matching -- which is what recovers
queries that hinge on notation (Z0, S21, "equation 4.48") that dense vectors
alone handle poorly.

    python scripts/ingest_qdrant.py [--limit N] [--batch 8] [--recreate]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "signal_integrity"
DENSE_DIM = 1024
MAX_LENGTH = 1024


def load_chunks(limit=None):
    rows = []
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import models

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from search_qdrant import get_client

    rows = load_chunks(args.limit)
    print(f"chunks: {len(rows)}")

    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    client = get_client()
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        for field in ("book", "chapter", "section"):
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
                if field != "chapter"
                else models.PayloadSchemaType.INTEGER,
            )

    total = 0
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        out = model.encode(
            [r["embed_text"] for r in batch],
            batch_size=len(batch),
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        points = []
        for row, dense, lex in zip(batch, out["dense_vecs"], out["lexical_weights"]):
            indices = [int(k) for k in lex.keys()]
            values = [float(v) for v in lex.values()]
            payload = {k: v for k, v in row.items() if k != "embed_text"}
            points.append(
                models.PointStruct(
                    id=row["id"],
                    vector={
                        "dense": dense.tolist(),
                        "sparse": models.SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        if start % (args.batch * 25) == 0 or total == len(rows):
            print(f"  {total}/{len(rows)}", flush=True)

    info = client.get_collection(COLLECTION)
    print(f"\ncollection '{COLLECTION}': {info.points_count} points at {QDRANT_PATH}")
    client.close()


if __name__ == "__main__":
    main()
