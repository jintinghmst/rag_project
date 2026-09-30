"""
Copy an embedded data/qdrant store into a Qdrant server, vectors and all.

The embedded store keeps every dense vector in one in-memory float64 array and
doubles that array when it grows, so it stops being viable well before this
corpus does: at ~158k points a query takes ~7 s (there is no ANN index, every
search is a full scan), and the next batch of documents fails outright trying to
allocate several contiguous gigabytes.

Moving to a server fixes both. The point of this script is that it moves the
*vectors*, not the text -- re-running the ingest against an empty server would
re-embed the whole corpus, which is hours of GPU time for work already done.

    docker compose up -d qdrant
    python rag.py migrate --to http://localhost:6333

Safe to re-run: points are upserted by id, so an interrupted migration continues
where it left off rather than duplicating anything.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import ingest_qdrant as iq  # noqa: E402

BATCH = 256


def source_client():
    """The local file store, read directly -- never via QDRANT_URL."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(corpus.QDRANT_PATH))


def dest_client(url, api_key):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=120)


def copy_collection(src, dst, name, make, quiet=False):
    """Stream every point of `name` from src to dst, vectors and payload intact."""
    from qdrant_client import models

    if not src.collection_exists(name):
        return 0
    if not dst.collection_exists(name):
        make()

    total, offset = 0, None
    while True:
        points, offset = src.scroll(collection_name=name, limit=BATCH, offset=offset,
                                    with_payload=True, with_vectors=True)
        if points:
            dst.upsert(collection_name=name, points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points])
            total += len(points)
            if not quiet:
                print(f"    {name}: {total}", flush=True)
        if offset is None:
            return total


def run(url, api_key=None, quiet=False):
    cfg = corpus.load()
    collection = os.environ.get("RAG_COLLECTION", cfg["collection"])
    manifest = iq.manifest_name(collection)

    if not corpus.QDRANT_PATH.exists():
        sys.exit(f"no local store at {corpus.QDRANT_PATH} -- nothing to migrate")

    src = source_client()
    dst = dest_client(url, api_key)
    try:
        if not quiet:
            info = src.get_collection(collection)
            print(f"  source: {info.points_count} points at {corpus.QDRANT_PATH}")
            print(f"  target: {url}")

        n = copy_collection(
            src, dst, collection,
            lambda: iq.ensure_collection(dst, collection), quiet)

        from qdrant_client import models
        m = copy_collection(
            src, dst, manifest,
            lambda: dst.create_collection(
                collection_name=manifest,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE)),
            quiet)

        served = dst.get_collection(collection).points_count
        return n, m, served
    finally:
        src.close()
        dst.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True, help="target server, e.g. http://localhost:6333")
    ap.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    args = ap.parse_args()

    points, manifest, served = run(args.to, args.api_key)
    print(f"\ncopied {points} points and {manifest} manifest entries")
    print(f"target now holds {served} points")
    print(f"\nSet these, then build or serve as usual:")
    print(f"  QDRANT_URL={args.to}")
    if args.api_key:
        print(f"  QDRANT_API_KEY=...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
