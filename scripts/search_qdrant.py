"""
Hybrid retrieval over the Qdrant store, with optional cross-encoder rerank.

Dense and sparse candidate lists are fused with reciprocal rank fusion inside
Qdrant, then (by default) reranked with BAAI/bge-reranker-v2-m3. On a corpus of
books that all cover the same subject in near-identical language, the reranker is
what separates the right chapter from a plausible one.

`search()` is importable and returns hits with full metadata -- that is the
function to wire into anything else.

    python scripts/search_qdrant.py "what causes far-end crosstalk?"
    python scripts/search_qdrant.py "skin effect resistance" --book hall_asi -k 5
    python scripts/search_qdrant.py "odd mode impedance" --no-rerank
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

_cfg = corpus.load()
QDRANT_PATH = corpus.QDRANT_PATH
MODEL_NAME = _cfg["embed_model"]
RERANKER_NAME = _cfg["rerank_model"]
COLLECTION = os.environ.get("RAG_COLLECTION", _cfg["collection"])

_model = None
_reranker = None


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
    return _model


def get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(RERANKER_NAME, use_fp16=True)
    return _reranker


def list_documents(client=None):
    """
    What the index actually holds: [{book, title, chunks}], newest state wins.

    Read from the manifest the ingest writes beside the collection rather than
    from corpus.json, so a server pointed at a shared Qdrant reports what is
    really searchable instead of what some other machine's registry says.
    """
    owns = client is None
    client = client or get_client()
    try:
        name = f"{COLLECTION}__manifest"
        if client.collection_exists(name):
            import ingest_qdrant

            points = ingest_qdrant.scroll_all(client, name)
            rows = [{"book": p.payload["key"], "title": p.payload.get("title", ""),
                     "chunks": p.payload.get("chunks", 0),
                     "kind": p.payload.get("kind", "book"),
                     "source": p.payload.get("source", "")} for p in points]
            if rows:
                return sorted(rows, key=lambda r: r["book"])
        return [{"book": k, "title": d["title"], "chunks": d.get("chunks", 0),
                 "kind": d.get("kind", "book"), "source": d.get("source", "")}
                for k, d in sorted(corpus.load()["documents"].items())]
    finally:
        if owns:
            client.close()


def get_client():
    """
    Shared Qdrant server if QDRANT_URL is set, else the local single-file store.

    The local store is a locked SQLite file -- one process at a time. Point
    QDRANT_URL at a Qdrant container to let a team share one index.
    """
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(QDRANT_PATH))


def search(query, k=5, candidates=30, book=None, rerank=True, client=None,
           source=None, kind=None):
    from qdrant_client import models

    owns_client = client is None
    if owns_client:
        client = get_client()

    try:
        out = get_model().encode(
            [query], return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        dense = out["dense_vecs"][0].tolist()
        lex = out["lexical_weights"][0]
        sparse = models.SparseVector(
            indices=[int(i) for i in lex.keys()],
            values=[float(v) for v in lex.values()],
        )

        # `source` matches a folder prefix ("IEEE Transactions on ... / 2025"),
        # so a whole journal or a single year can be selected without naming
        # every paper in it
        must = []
        if book:
            must.append(models.FieldCondition(key="book",
                                              match=models.MatchValue(value=book)))
        if kind:
            must.append(models.FieldCondition(key="kind",
                                              match=models.MatchValue(value=kind)))
        if source:
            must.append(models.FieldCondition(key="source",
                                              match=models.MatchText(text=source)))
        flt = models.Filter(must=must) if must else None

        res = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=candidates, filter=flt),
                models.Prefetch(query=sparse, using="sparse", limit=candidates, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidates if rerank else k,
            with_payload=True,
        ).points

        hits = [{"score": p.score, **p.payload} for p in res]

        if rerank and hits:
            scores = get_reranker().compute_score(
                [[query, h["text"]] for h in hits], normalize=True
            )
            if not isinstance(scores, list):
                scores = [scores]
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)

        return hits[:k]
    finally:
        if owns_client:
            client.close()


def cite(h):
    """
    One line a reader can follow back to the page.

    A paper names its venue and year, which a textbook does not need -- "Hall &
    Heck" identifies itself, while a paper title alone leaves the reader unable
    to tell a 2026 result from a 2022 one.
    """
    sec = f"S{h['section']} {h['section_title']}" if h.get("section") else h.get("section_title", "")
    pages = (
        f"p.{h['page_start']}"
        if h.get("page_start") == h.get("page_end")
        else f"pp.{h.get('page_start')}-{h.get('page_end')}"
    )
    parts = [h["book_title"]]
    if h.get("source"):
        parts.append(h["source"])
    parts += [sec, pages]
    return " | ".join(p for p in parts if p)


def main():
    # the corpus is full of math symbols; a cp936/cp1252 console raises
    # UnicodeEncodeError mid-print and truncates the results
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--book", default=None, help="restrict to one document key")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--full", action="store_true", help="print the whole chunk")
    args = ap.parse_args()

    hits = search(args.query, k=args.k, candidates=args.candidates,
                  book=args.book, rerank=args.rerank)

    for i, h in enumerate(hits, 1):
        score = h.get("rerank_score", h["score"])
        print(f"\n[{i}] {score:.4f}  {cite(h)}")
        text = h["text"] if args.full else h["text"][:600].replace("\n", " ")
        print(f"    {text}{'' if args.full else ' ...'}")


if __name__ == "__main__":
    main()
