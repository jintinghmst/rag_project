"""
Hybrid retrieval over the local Qdrant store, with optional cross-encoder rerank.

Dense and sparse candidate lists are fused with reciprocal rank fusion inside
Qdrant, then (by default) reranked with BAAI/bge-reranker-v2-m3. On a corpus of
three books that all cover crosstalk, impedance and loss in near-identical
language, the reranker is what separates the right chapter from a plausible one.

    python scripts/search_qdrant.py "what causes far-end crosstalk?"
    python scripts/search_qdrant.py "skin effect resistance" --book hall_asi -k 5
    python scripts/search_qdrant.py "odd mode impedance" --no-rerank
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = ROOT / "data" / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
COLLECTION = "signal_integrity"

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


def search(query, k=5, candidates=30, book=None, rerank=True, client=None):
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

        flt = (
            models.Filter(must=[models.FieldCondition(key="book", match=models.MatchValue(value=book))])
            if book
            else None
        )

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
    sec = f"S{h['section']} {h['section_title']}" if h.get("section") else h.get("section_title", "")
    pages = (
        f"p.{h['page_start']}"
        if h.get("page_start") == h.get("page_end")
        else f"pp.{h.get('page_start')}-{h.get('page_end')}"
    )
    return f"{h['book_title']} | {sec} | {pages}"


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
    ap.add_argument("--book", default=None, help="paul_mtl | hall_asi | dsi_mod")
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
