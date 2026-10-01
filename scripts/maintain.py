"""
Integrity checking and backup, so the 42 GB of source PDFs need not stay forever.

The pipeline's real source of truth is not `original/`. Once a document has been
extracted, every later stage runs from `extracted/<key>.txt` -- 475 MB for the
whole corpus against 42 GB of PDFs, about 1%. Delete the PDFs and the build still
cleans, chunks, embeds and serves; what you lose is the ability to *re*-extract,
which matters only if the extraction code changes or you want a different engine.

So two commands:

    rag.py verify     does the chain still hold -- registry, text, chunks, index?
    rag.py backup     copy what cannot be recomputed somewhere safe

What a backup must contain is small:

    corpus.json       9 MB    titles, page ranges, slots -- the expensive scan
    extracted/        475 MB  the text; re-deriving it needs the PDFs back
    (index snapshot)  ~7 GB   optional: skips ~4 h of GPU time on restore

The first two are the ones that cannot be recovered without the originals. The
index is merely expensive, not irreplaceable -- it rebuilds from them.
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402


# ---------------------------------------------------------------- integrity

def verify(deep=False, quiet=False):
    """
    Walk the chain registry -> text -> chunks -> index and report every break.

    Returns (summary, problems, pending). Work still queued -- a document chunked
    but not yet embedded -- is *pending*, not a problem: during a build that is
    thousands of documents, and reporting them as faults would bury the handful
    that are genuinely broken. `deep` also counts points per document, which is
    slow but catches a document whose manifest entry outlived its vectors.
    """
    cfg = corpus.load()
    docs = cfg["documents"]
    summary = {"documents": len(docs), "text": 0, "clean": 0, "chunks": 0,
               "indexed": 0, "stale_text": 0, "missing_pdf": 0}
    problems, pending = [], []

    indexed = {}
    try:
        import search_qdrant as sq
        for row in sq.list_documents():
            indexed[row["book"]] = row.get("chunks", 0)
        summary["index"] = f"{sq.COLLECTION} ({len(indexed)} documents)"
    except Exception as exc:
        summary["index"] = f"UNREACHABLE: {type(exc).__name__}: {exc}"

    for key, doc in docs.items():
        paths = corpus.doc_paths(key, doc)
        if not paths["pdf"] or not paths["pdf"].exists():
            summary["missing_pdf"] += 1

        if paths["text"].exists():
            summary["text"] += 1
            # a recorded hash that no longer matches means the text changed under
            # the build's feet; everything downstream of it is suspect
            if doc.get("text_hash") and corpus.file_hash(paths["text"]) != doc["text_hash"]:
                summary["stale_text"] += 1
                problems.append((key, "text", "content differs from recorded hash"))
        elif paths["pdf"] and paths["pdf"].exists():
            pending.append((key, "text", "registered, not extracted yet"))
        else:
            problems.append((key, "text", "no extracted text and no PDF to redo it"))

        if paths["clean"].exists():
            summary["clean"] += 1
        if paths["chunks"].exists():
            summary["chunks"] += 1
        elif paths["text"].exists():
            pending.append((key, "chunks", "text present, not chunked yet"))

        if key in indexed:
            summary["indexed"] += 1
            if paths["chunks"].exists() and indexed[key] == 0:
                problems.append((key, "index", "in the manifest but holds no chunks"))
        elif paths["chunks"].exists():
            pending.append((key, "index", "chunked, not embedded yet"))

    if deep and indexed:
        problems += _deep_check(indexed, quiet)
    return summary, problems, pending


def _deep_check(indexed, quiet):
    """Count real points per document; a manifest can outlive the vectors."""
    from qdrant_client import models

    import search_qdrant as sq

    out = []
    client = sq.get_client()
    try:
        for n, (key, claimed) in enumerate(sorted(indexed.items()), 1):
            if not quiet and n % 500 == 0:
                print(f"    deep check {n}/{len(indexed)}", flush=True)
            actual = client.count(
                collection_name=sq.COLLECTION,
                count_filter=models.Filter(must=[models.FieldCondition(
                    key="book", match=models.MatchValue(value=key))]),
                exact=True).count
            if actual != claimed:
                out.append((key, "index", f"manifest says {claimed}, index holds {actual}"))
    finally:
        client.close()
    return out


# ------------------------------------------------------------------- backup

def _sha256(path, block=4 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive(src_dir, dest, quiet=False):
    """tar.gz a directory; text compresses to roughly a quarter of its size."""
    if not quiet:
        print(f"  archiving {src_dir.name}/", flush=True)
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return dest


def snapshot_index(dest, quiet=False):
    """
    Ask Qdrant for a snapshot of the collection and download it.

    Only works against a server; the embedded store has no snapshot API. The file
    restores into any Qdrant, which is what makes it worth keeping despite the
    size -- the alternative is re-embedding the corpus.
    """
    import search_qdrant as sq

    url = os.environ.get("QDRANT_URL")
    if not url:
        return None, "QDRANT_URL is not set (no server to snapshot)"

    base = f"{url.rstrip('/')}/collections/{sq.COLLECTION}/snapshots"
    key = os.environ.get("QDRANT_API_KEY")
    headers = {"api-key": key} if key else {}

    try:
        req = urllib.request.Request(base, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=3600) as r:
            name = json.load(r)["result"]["name"]
        if not quiet:
            print(f"  downloading snapshot {name}", flush=True)
        req = urllib.request.Request(f"{base}/{name}", headers=headers)
        with urllib.request.urlopen(req, timeout=3600) as r, open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh, 8 << 20)
        # the server keeps its own copy too; drop it so storage does not creep
        urllib.request.urlopen(urllib.request.Request(
            f"{base}/{name}", method="DELETE", headers=headers), timeout=600)
        return dest, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def backup(dest_root, with_index=False, keep=3, quiet=False):
    """Write a dated, self-describing backup; prune all but the newest `keep`."""
    dest_root = Path(dest_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = dest_root / f"rag-backup-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    artefacts = {}
    shutil.copy2(corpus.REGISTRY, out / "corpus.json")
    artefacts["corpus.json"] = _sha256(out / "corpus.json")

    text_tar = _archive(corpus.TEXT_DIR, out / "extracted.tar.gz", quiet)
    artefacts["extracted.tar.gz"] = _sha256(text_tar)

    note = None
    if with_index:
        snap, err = snapshot_index(out / "index.snapshot", quiet)
        if snap:
            artefacts["index.snapshot"] = _sha256(snap)
        else:
            note = f"index snapshot skipped: {err}"

    summary, problems, pending = verify(quiet=True)
    (out / "BACKUP.json").write_text(json.dumps({
        "created": stamp,
        "documents": summary["documents"],
        "indexed": summary["indexed"],
        "problems": len(problems),
        "pending": len(pending),
        "note": note,
        "sha256": artefacts,
        "restore": [
            "copy corpus.json back to the project root",
            "tar -xzf extracted.tar.gz into the project root",
            "python rag.py build   # re-cleans, re-chunks, re-embeds from the text",
            "or, with index.snapshot, upload it to Qdrant and skip the embedding",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    pruned = _prune(dest_root, keep)
    return out, size, note, pruned


def _prune(root, keep):
    old = sorted(root.glob("rag-backup-*"), key=lambda p: p.name, reverse=True)[keep:]
    for path in old:
        shutil.rmtree(path, ignore_errors=True)
    return [p.name for p in old]
