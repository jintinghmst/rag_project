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


def extract_pymupdf(pdf, span=None, handle=None):
    import pymupdf

    doc = handle or pymupdf.open(pdf)
    try:
        first, last = span or (1, doc.page_count)
        for i in range(max(1, first), min(last, doc.page_count) + 1):
            yield i, doc[i - 1].get_text("text")
    finally:
        if handle is None:
            doc.close()


def extract_pdftotext(pdf, span=None, handle=None):
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not on PATH (install poppler, or use --engine pymupdf)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        cmd = ["pdftotext", "-layout"]
        if span:
            cmd += ["-f", str(span[0]), "-l", str(span[1])]
        subprocess.run(cmd + [str(pdf), str(out)], check=True, capture_output=True)
        # pdftotext separates pages with a form feed
        start = span[0] if span else 1
        for i, page in enumerate(
                out.read_text(encoding="utf-8", errors="replace").split("\f"), start):
            yield i, page


ENGINES = {"pymupdf": extract_pymupdf, "pdftotext": extract_pdftotext}


def extract(pdf, engine="pymupdf", span=None, page_offset=None, handle=None, label=None):
    """
    Text for one document, which may be a page range inside a larger PDF.

    `[PAGE n]` carries the number a reader would cite. For an article cut out of
    a journal issue that is the printed folio, recovered via `page_offset`; where
    that could not be determined confidently the PDF's own index is used, so a
    citation is never silently wrong -- only sometimes less convenient.
    """
    parts, chars, pages = [], 0, 0
    for n, text in ENGINES[engine](pdf, span, handle):
        body = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
        printed = n + page_offset if page_offset is not None else n
        parts.append(f"[PAGE {printed}]\n{body}\n")
        chars += len(body.strip())
        pages += 1
    scope = f" pages {span[0]}-{span[1]}" if span else ""
    header = (
        f"# SOURCE: {pdf.name}{scope}\n"
        + (f"# TITLE: {label}\n" if label else "")
        + f"# PAGES: {'printed folio' if page_offset is not None else 'pdf index'}\n"
        f"# EXTRACTION: {engine}. Equations are best-effort inline text, not verified LaTeX.\n\n"
    )
    return header + "\n".join(parts), pages, chars


def run(keys=None, engine="pymupdf", force=False, quiet=False):
    """
    Extract every registered document whose cached text is missing or stale.

    Work is grouped by source PDF, not by document. A journal issue holds ~60
    articles, and opening and hashing a 700 MB file once per article instead of
    once per issue is the difference between minutes and hours.
    """
    cfg = corpus.load()
    # registration is its own stage now (scan_pdfs.py): an issue PDF becomes many
    # documents, which needs the PDF opened, not just its filename read
    added = []
    corpus.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    wanted = [(k, d) for k, d in cfg["documents"].items() if not keys or k in keys]
    by_pdf = {}
    for key, doc in wanted:
        by_pdf.setdefault(doc.get("pdf"), []).append((key, doc))

    for n, (rel, group) in enumerate(sorted(by_pdf.items(), key=lambda kv: kv[0] or ""), 1):
        pdf = corpus.doc_paths(group[0][0], group[0][1])["pdf"]
        if pdf is None or not pdf.exists():
            for key, doc in group:
                if corpus.doc_paths(key, doc)["text"].exists():
                    report.append((key, "no pdf", "using cached text"))
                else:
                    report.append((key, "MISSING", f"no {doc.get('pdf')} and no cached text"))
            continue

        pending = []
        for key, doc in group:
            paths = corpus.doc_paths(key, doc)
            if not force and paths["text"].exists() and doc.get("pdf_hash"):
                report.append((key, "cached", paths["text"].name))
                continue
            pending.append((key, doc, paths))
        if not pending:
            continue

        digest = corpus.file_hash(pdf)          # hashed once for the whole issue
        still = []
        for key, doc, paths in pending:
            if not force and paths["text"].exists() and doc.get("pdf_hash") == digest:
                report.append((key, "cached", paths["text"].name))
            elif not force and paths["text"].exists() and "pdf_hash" not in doc:
                # text that predates hash tracking (or was produced by hand, e.g.
                # OCR) is trusted rather than thrown away
                doc["pdf_hash"] = digest
                report.append((key, "adopted", f"kept existing {paths['text'].name}"))
            else:
                still.append((key, doc, paths))
        if not still:
            continue

        if not quiet:
            print(f"  [{n}/{len(by_pdf)}] {pdf.name[:58]} -> {len(still)} document(s)",
                  flush=True)

        handle = None
        try:
            if engine == "pymupdf":
                import pymupdf
                handle = pymupdf.open(pdf)      # opened once for every article
            for key, doc, paths in still:
                span = tuple(doc["pages"]) if doc.get("pages") else None
                text, pages, chars = extract(
                    pdf, engine, span=span, page_offset=doc.get("page_offset"),
                    handle=handle, label=doc.get("title"))
                per_page = chars / max(pages, 1)
                if per_page < OCR_THRESHOLD:
                    report.append((key, "NO TEXT LAYER",
                                   f"{per_page:.0f} chars/page -- OCR it (ocrmypdf) first"))
                    continue
                paths["text"].write_text(text, encoding="utf-8")
                doc["pdf_hash"] = digest
                doc["engine"] = engine
                report.append((key, "extracted", f"{pages} pages, {chars // 1000}k chars"))
        finally:
            if handle is not None:
                handle.close()

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
