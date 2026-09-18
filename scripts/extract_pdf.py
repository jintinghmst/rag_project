"""
Stage 1: original/*.pdf -> extracted/<key>.txt, one `[PAGE n]` marker per page.

Extraction is cached on the PDF's content hash, because it is the slow, noisy
step and its output is what every later stage was tuned against: re-running the
build does not re-extract a book whose PDF has not changed.

Two engines. PyMuPDF is the default -- it is a pip wheel, so a fresh clone on any
OS can extract without installing poppler. `--engine pdftotext` uses poppler's
`pdftotext -layout` instead, which keeps tables and equations in better column
order when it happens to be installed.

A PDF with no text layer (a scan) yields almost nothing from either engine and is
reported as needing OCR rather than being silently indexed as empty.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

# below this many characters per page, the PDF is a scan with no text layer
OCR_THRESHOLD = 120


def extract_pymupdf(pdf):
    import pymupdf

    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            yield i, page.get_text("text")


def extract_pdftotext(pdf):
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not on PATH (install poppler, or use --engine pymupdf)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)],
                       check=True, capture_output=True)
        # pdftotext separates pages with a form feed
        for i, page in enumerate(out.read_text(encoding="utf-8", errors="replace").split("\f"), 1):
            yield i, page


ENGINES = {"pymupdf": extract_pymupdf, "pdftotext": extract_pdftotext}


def extract(pdf, engine="pymupdf"):
    parts, chars, pages = [], 0, 0
    for n, text in ENGINES[engine](pdf):
        body = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
        parts.append(f"[PAGE {n}]\n{body}\n")
        chars += len(body.strip())
        pages += 1
    header = (
        f"# SOURCE: {pdf.name}\n"
        f"# EXTRACTION: {engine}. Equations are best-effort inline text, not verified LaTeX.\n\n"
    )
    return header + "\n".join(parts), pages, chars


def run(keys=None, engine="pymupdf", force=False, quiet=False):
    """Extract every registered document whose cached text is missing or stale."""
    cfg = corpus.load()
    added = corpus.discover(cfg)
    corpus.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    for key, doc in cfg["documents"].items():
        if keys and key not in keys:
            continue
        paths = corpus.doc_paths(key, doc)
        pdf = paths["pdf"]
        if pdf is None or not pdf.exists():
            if paths["text"].exists():
                report.append((key, "no pdf", "using cached text"))
                continue
            report.append((key, "MISSING", f"no {doc.get('pdf')} and no cached text"))
            continue

        digest = corpus.file_hash(pdf)
        if not force and paths["text"].exists() and doc.get("pdf_hash") == digest:
            report.append((key, "cached", paths["text"].name))
            continue
        # text that predates hash tracking (or was produced by hand, e.g. OCR)
        # is trusted rather than thrown away -- record its hash and move on
        if not force and paths["text"].exists() and "pdf_hash" not in doc:
            doc["pdf_hash"] = digest
            report.append((key, "adopted", f"kept existing {paths['text'].name}"))
            continue

        if not quiet:
            print(f"  extracting {key} <- {pdf.name}", flush=True)
        text, pages, chars = extract(pdf, engine)
        per_page = chars / max(pages, 1)
        if per_page < OCR_THRESHOLD:
            report.append((key, "NO TEXT LAYER",
                           f"{per_page:.0f} chars/page -- run OCR (ocrmypdf) on the PDF first"))
            continue
        paths["text"].write_text(text, encoding="utf-8")
        doc["pdf_hash"] = digest
        doc["engine"] = engine
        report.append((key, "extracted", f"{pages} pages, {chars // 1000}k chars"))

    corpus.save(cfg)
    return cfg, report, added


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="pymupdf")
    ap.add_argument("--only", nargs="*", help="document keys to extract")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    _, report, added = run(args.only, args.engine, args.force)
    for key in added:
        print(f"+ registered new document: {key}")
    w = max((len(k) for k, _, _ in report), default=4)
    for key, status, detail in report:
        print(f"{key:<{w}}  {status:<14} {detail}")
    if any(s in {"MISSING", "NO TEXT LAYER"} for _, s, _ in report):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
