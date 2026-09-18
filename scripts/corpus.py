"""
The document registry: what is in the corpus, and where each stage's files live.

Everything else in the pipeline reads the corpus from here, so adding a book is
dropping a PDF into `original/` -- nothing in the scripts is per-book.

`corpus.json` holds one entry per document, keyed by a short `key` used as the
Qdrant payload filter, the citation label, and the chunk-id namespace. New PDFs
are registered automatically on the next build with values derived from the
filename plus auto-detected front/back matter; the file is then yours to edit,
and a hand-set value is never overwritten (`auto` lists the fields that were
guessed and may be re-guessed).
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "corpus.json"

PDF_DIR = ROOT / "original"       # drop new PDFs here
TEXT_DIR = ROOT / "extracted"     # stage 1: raw text + [PAGE n] markers (cached)
CLEAN_DIR = ROOT / "data" / "clean"
CHUNK_DIR = ROOT / "data" / "chunks"
QDRANT_PATH = ROOT / "data" / "qdrant"

# Each document owns a contiguous block of chunk ids, so adding or rebuilding one
# book never renumbers another. read_context() walks neighbours by id, which only
# holds because ids are dense and in reading order within a block.
ID_STRIDE = 1 << 20              # 1,048,576 chunks per document

DEFAULTS = {
    "collection": "signal_integrity",
    "embed_model": "BAAI/bge-m3",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    # per-page publisher furniture, dropped from every document
    "banner_patterns": [r"Downloaded from https://onlinelibrary\.wiley\.com"],
    "documents": {},
}

PAGE_RE = re.compile(r"^\[PAGE (\d+)\]$")


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def short_key(stem):
    """A terse key from a filename: first two meaningful words, e.g. `paul_mtl`."""
    words = [w for w in slug(stem).split("_") if len(w) > 2 and not w.isdigit()]
    drop = {"the", "and", "with", "for", "series", "edition", "wiley", "press",
            "ieee", "inc", "compress", "analysis", "vol"}
    words = [w for w in words if w not in drop] or [slug(stem)[:12] or "doc"]
    return "_".join(words[:2])[:24]


def load():
    cfg = dict(DEFAULTS)
    if REGISTRY.exists():
        cfg.update(json.loads(REGISTRY.read_text(encoding="utf-8")))
    cfg.setdefault("documents", {})
    return cfg


def save(cfg):
    REGISTRY.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def pdfs():
    return sorted(p for p in PDF_DIR.glob("*.pdf") if not p.name.startswith("~"))


def next_slot(cfg):
    used = {d.get("slot", -1) for d in cfg["documents"].values()}
    n = 0
    while n in used:
        n += 1
    return n


def discover(cfg):
    """Register any PDF in original/ that the registry does not know about yet."""
    known = {d.get("pdf") for d in cfg["documents"].values()}
    added = []
    for pdf in pdfs():
        if pdf.name in known:
            continue
        key = short_key(pdf.stem)
        while key in cfg["documents"]:
            key += "_2"
        cfg["documents"][key] = {
            "pdf": pdf.name,
            "title": re.sub(r"\s+", " ", pdf.stem).strip(),
            "slot": next_slot(cfg),
            "text": f"{key}.txt",
            "skip_pages": [],
            "body_end_page": None,
            "auto": ["title", "skip_pages", "body_end_page"],
        }
        added.append(key)
    return added


def doc_paths(key, doc):
    return {
        "pdf": PDF_DIR / doc["pdf"] if doc.get("pdf") else None,
        "text": TEXT_DIR / (doc.get("text") or f"{key}.txt"),
        "clean": CLEAN_DIR / f"{key}.txt",
        "chunks": CHUNK_DIR / f"{key}.jsonl",
    }


def id_range(doc):
    base = doc["slot"] * ID_STRIDE
    return base, base + ID_STRIDE


def file_hash(*paths):
    """
    Content hash used to decide what a build can skip.

    Text files are normalized to LF first. Git rewrites line endings on checkout,
    so hashing raw bytes would make every hash in corpus.json machine-specific and
    force a full rebuild on a fresh clone -- and leave a diff behind afterwards.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in paths:
        p = Path(p) if p else None
        if not p or not p.exists():
            continue
        data = p.read_bytes()
        if p.suffix.lower() in {".txt", ".json", ".jsonl", ".md"}:
            data = data.replace(b"\r\n", b"\n")
        h.update(data)
    return h.hexdigest()


def split_pages(text):
    """[(page_number|None, [lines])] in document order, keeping the marker out."""
    pages, cur, num = [], [], None
    for line in text.split("\n"):
        m = PAGE_RE.match(line.strip())
        if m:
            pages.append((num, cur))
            num, cur = int(m.group(1)), []
        else:
            cur.append(line)
    pages.append((num, cur))
    return [p for p in pages if p[0] is not None or any(l.strip() for l in p[1])]


# ---- front / back matter detection -----------------------------------------
# Front matter (a table of contents) and back matter (the index) are pure noise
# for retrieval and actively harmful: a contents page is a dense list of every
# term in the book and ranks highly for almost any query. The detectors below
# are heuristics whose verdict is written into corpus.json for inspection.

_FOLIO_END = re.compile(r"[\s.]\d{1,4}$")
_INDEX_ENTRY = re.compile(r"[A-Za-z].*?,\s*[\d,\s\-–]+$")


def _lines(page):
    return [l.strip() for l in page[1] if l.strip()]


def _contents_like(page):
    ls = _lines(page)
    if len(ls) < 6:
        return False
    if re.fullmatch(r"(TABLE OF )?CONTENTS", ls[0], re.I):
        return True
    short = sum(len(l) < 80 for l in ls) / len(ls)
    folio = sum(bool(_FOLIO_END.search(l)) for l in ls) / len(ls)
    return short > 0.85 and folio > 0.3


def _index_like(page):
    ls = _lines(page)
    if not ls:
        return False
    if re.fullmatch(r"INDEX|Index(\s+Terms)?", ls[0]):
        return True          # an explicit heading settles it, however short
    if len(ls) < 8:
        return False
    return sum(bool(_INDEX_ENTRY.match(l)) for l in ls) / len(ls) > 0.45


def _body_like(page):
    """Running prose: the signal that a contents block has ended."""
    ls = _lines(page)
    return sum(len(l) > 88 for l in ls) >= 3


def detect_front_matter(pages):
    """Page ranges to drop: the contents block, if one sits in the first 15%."""
    limit = max(4, int(len(pages) * 0.15))
    head = [p for p in pages[:limit] if p[0] is not None]
    hits = {p[0] for p in head if _contents_like(p)}
    if not hits:
        return []
    by_num = {p[0]: p for p in head}

    # A contents block is contiguous, but every other page of one often fails the
    # ratio test (a part-title page, a roman folio). Grow the run across gaps of
    # up to two pages, then across trailing pages until real prose starts -- the
    # last contents page rarely looks like one.
    lo = min(hits)
    hi = lo
    for n in sorted(by_num):
        if n <= hi:
            continue
        if n in hits:
            hi = n
        elif n - hi > 2:
            break
    for n in sorted(by_num):
        if n <= hi:
            continue
        if n - hi > 3 or _body_like(by_num[n]):
            break
        hi = n
    return [[lo, hi]]


def detect_back_matter(pages):
    """The first page of the index, if the tail of the book looks like one."""
    numbered = [p for p in pages if p[0] is not None]
    if not numbered:
        return None
    tail = numbered[int(len(numbered) * 0.8):]
    # Walk back from the end: the index is the last long run of index-like pages.
    # Scanning forwards instead trips over the odd table-heavy page in an
    # appendix, and over the "about the author" page that often follows the index.
    start = end = None
    gap = 0
    for page in reversed(tail):
        if _index_like(page):
            start, gap = page[0], 0
            if end is None:
                end = page[0]
        elif start is not None:
            gap += 1
            if gap > 2:
                break
    if start is None or end - start < 2:
        return None
    return start


def detect_matter(text):
    pages = split_pages(text)
    return detect_front_matter(pages), detect_back_matter(pages)
