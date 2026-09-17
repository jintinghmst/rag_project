"""
Section-aware chunking of the cleaned textbooks into data/chunks.jsonl.

Chunks never cross a section boundary. Within a section, paragraphs are packed
up to TARGET_TOKENS (measured with the bge-m3 tokenizer) with a paragraph-level
overlap, so a derivation and the equation it refers to tend to stay together.

Each chunk carries book / chapter / section / page-range metadata for citation
and for metadata-filtered retrieval. The embedded text is prefixed with a short
provenance header, which measurably helps retrieval on a corpus where three
books discuss the same concepts in near-identical language.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "clean"
OUT = ROOT / "data" / "chunks.jsonl"

MODEL_NAME = "BAAI/bge-m3"
TARGET_TOKENS = 450
MAX_TOKENS = 800
MIN_TOKENS = 40
OVERLAP_TOKENS = 80

BOOKS = {
    "clayton_paul_multiconductor_transmission_lines.txt": {
        "short": "Paul, Analysis of Multiconductor Transmission Lines (2e)",
        "key": "paul_mtl",
    },
    "hall_advanced_signal_integrity.txt": {
        "short": "Hall & Heck, Advanced Signal Integrity for High-Speed Digital Designs",
        "key": "hall_asi",
    },
    "digital_signal_integrity_modeling_simulation.txt": {
        "short": "Digital Signal Integrity: Modeling and Simulation with Interconnects and Packages",
        "key": "dsi_mod",
    },
}

PAGE_RE = re.compile(r"^\[PAGE (\d+)\]$")
# "3.4.6 Field Mapping" / "1.2 THE PROBLEM" -- but not a contents line ending in a
# folio, and not a numeric data row that happens to start "83.61 ..."
HEADING_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?\s+([A-Z].{3,90})$")
MAX_CHAPTER = 20


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:  # fall back to a rough estimate so chunking still runs
        print(f"! tokenizer unavailable ({exc.__class__.__name__}); using word-count estimate")
        return None


class Counter:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text):
        if self.tok is None:
            return int(len(text.split()) * 1.35) + 1
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def is_heading(line):
    m = HEADING_RE.match(line.strip())
    if not m:
        return None
    chapter = int(m.group(1))
    if chapter > MAX_CHAPTER or chapter == 0:
        return None
    title = m.group(4).strip()
    # contents lines carry a trailing page number: "1.1 Tradeoffs ... 2"
    if re.search(r"\s\d{1,4}$", title):
        return None
    if title.endswith(",") or sum(c.isalpha() for c in title) < 4:
        return None
    return {
        "chapter": chapter,
        "section": f"{m.group(1)}.{m.group(2)}" + (f".{m.group(3)}" if m.group(3) else ""),
        "title": re.sub(r"\s+", " ", title),
    }


def parse_sections(path):
    """Yield {chapter, section, title, blocks:[(text, page_start, page_end)]}."""
    page = None
    current = {"chapter": None, "section": None, "title": "front matter", "blocks": []}
    buf, buf_start = [], None
    sections = []

    def flush_para():
        nonlocal buf, buf_start
        if buf:
            text = "\n".join(buf).strip()
            if text:
                current["blocks"].append((text, buf_start, page))
        buf, buf_start = [], None

    def flush_section():
        flush_para()
        if current["blocks"]:
            sections.append(dict(current))

    for raw in path.read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip()
        if line.startswith("#"):
            continue
        m = PAGE_RE.match(line.strip())
        if m:
            page = int(m.group(1))
            continue
        h = is_heading(line)
        if h:
            flush_section()
            current = dict(h, blocks=[])
            continue
        if not line.strip():
            flush_para()
            continue
        if buf_start is None:
            buf_start = page
        buf.append(line)
    flush_section()
    return sections


def pack(blocks, count, header):
    """Greedily pack paragraphs into chunks, overlapping the tail of the previous."""
    chunks, cur, cur_tokens = [], [], 0
    head_tokens = count(header)

    def emit():
        if not cur:
            return
        text = "\n\n".join(b[0] for b in cur)
        pages = [p for b in cur for p in (b[1], b[2]) if p is not None]
        chunks.append({
            "text": text,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "n_tokens": head_tokens + count(text),
        })

    for block in blocks:
        n = count(block[0])
        if n > MAX_TOKENS:
            # an oversized paragraph (usually a code listing or a table dump):
            # emit what we have, then split it on sentence boundaries
            emit()
            cur, cur_tokens = [], 0
            for piece in split_long(block[0], count):
                chunks.append({
                    "text": piece,
                    "page_start": block[1],
                    "page_end": block[2],
                    "n_tokens": head_tokens + count(piece),
                })
            continue
        if cur_tokens + n > TARGET_TOKENS and cur:
            emit()
            tail, tail_tokens = [], 0
            for b in reversed(cur):
                bn = count(b[0])
                if tail_tokens + bn > OVERLAP_TOKENS:
                    break
                tail.insert(0, b)
                tail_tokens += bn
            cur, cur_tokens = list(tail), tail_tokens
        cur.append(block)
        cur_tokens += n
    emit()
    return chunks


def split_long(text, count):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur, cur_n = [], [], 0
    for s in sentences:
        n = count(s)
        if cur and cur_n + n > TARGET_TOKENS:
            out.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return [o for o in out if o.strip()]


def main():
    count = Counter(load_tokenizer())
    records, cid = [], 0
    stats = {}

    for fname, meta in BOOKS.items():
        sections = parse_sections(SRC / fname)
        n_book = 0
        for sec in sections:
            label = f"Section {sec['section']} {sec['title']}" if sec["section"] else sec["title"]
            header = f"[{meta['short']} | {label}]"
            for ch in pack(sec["blocks"], count, header):
                if ch["n_tokens"] < MIN_TOKENS:
                    continue
                records.append({
                    "id": cid,
                    "book": meta["key"],
                    "book_title": meta["short"],
                    "chapter": sec["chapter"],
                    "section": sec["section"],
                    "section_title": sec["title"],
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "n_tokens": ch["n_tokens"],
                    "text": ch["text"],
                    "embed_text": f"{header}\n{ch['text']}",
                })
                cid += 1
                n_book += 1
        stats[meta["key"]] = (len(sections), n_book)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    toks = [r["n_tokens"] for r in records]
    toks.sort()
    print(f"{'book':<12} {'sections':>9} {'chunks':>8}")
    for k, (s, c) in stats.items():
        print(f"{k:<12} {s:>9} {c:>8}")
    print(f"\ntotal chunks : {len(records)}")
    print(f"tokens       : min {toks[0]}  p50 {toks[len(toks)//2]}  "
          f"p95 {toks[int(len(toks)*0.95)]}  max {toks[-1]}")
    print(f"written      : {OUT}")


if __name__ == "__main__":
    main()
