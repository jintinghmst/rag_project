"""
Clean the three signal-integrity textbook .txt extractions for RAG ingestion.

Per book:
  1. drop front matter (table of contents) and back matter (index)
  2. strip the per-page Wiley download banner
  3. strip running headers / footers and bare folio numbers around [PAGE n]
  4. normalize unicode (ligatures, smart quotes, nbsp, soft hyphen)
  5. repair line-break hyphenation, using corpus evidence to decide whether the
     hyphen is real ("high-speed") or a soft wrap hyphen ("propaga-tion")
  6. reflow hard-wrapped prose into paragraphs, leaving equation lines alone
  7. drop pure-punctuation junk lines and collapse blank runs

[PAGE n] markers are preserved: the chunker uses them for page citations and
strips them before embedding.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "training"
DST = ROOT / "data" / "clean"

# line ranges are 1-based and inclusive, verified by inspection of each file
BOOKS = {
    "clayton_paul_multiconductor_transmission_lines.txt": {
        "drop_ranges": [(68, 528)],   # table of contents
        "truncate_at": 49605,         # [PAGE 796] -> INDEX
    },
    "hall_advanced_signal_integrity.txt": {
        "drop_ranges": [(68, 484)],   # table of contents
        "truncate_at": 45532,         # [PAGE 662] -> INDEX
    },
    "digital_signal_integrity_modeling_simulation.txt": {
        "drop_ranges": [],            # no front matter in the extraction
        "truncate_at": 21510,         # [PAGE 536] -> Index
    },
}

PAGE_RE = re.compile(r"^\[PAGE (\d+)\]$")
BANNER_RE = re.compile(r"Downloaded from https://onlinelibrary\.wiley\.com")
ROMAN_RE = re.compile(r"^[ivxlcdm]{1,7}$", re.I)
FOLIO_RE = re.compile(r"^\d{1,4}$")
# "520 Sample Layer Peeling Code Appendix E" / "2 Signal Integrity Chapter 5"
OCR_HEADER_RE = re.compile(
    r"^\d{1,4}\s+.{0,60}?\s+(Chapter\s+\d+|Appendix\s+[A-Z])$|"
    r"^(Chapter\s+\d+|Appendix\s+[A-Z]\s+.{0,60}?\s+\d{1,4})$|"
    r"^Section\s+\d+\.\d+\..{0,60}?\s+\d{1,4}$"
)

TRANSLATE = {
    0x00a0: " ", 0x2007: " ", 0x202f: " ", 0x2009: " ", 0x200a: " ", 0x2002: " ",
    0x2003: " ", 0x00ad: None, 0x200b: None, 0xfeff: None,
    0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"', 0x2032: "'",
    0x2013: "-", 0x2014: "-", 0x2010: "-", 0x2011: "-",
    0xfb00: "ff", 0xfb01: "fi", 0xfb02: "fl", 0xfb03: "ffi", 0xfb04: "ffl",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return text.translate(TRANSLATE)


def alpha_ratio(s):
    stripped = s.replace(" ", "")
    if not stripped:
        return 0.0
    return sum(c.isalpha() for c in stripped) / len(stripped)


def is_prose(s):
    """A line of running text, as opposed to an equation fragment or a heading."""
    t = s.strip()
    if len(t) < 25 or " " not in t or alpha_ratio(t) < 0.72:
        return False
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return False  # a running header or section title, not running text
    return True


def find_running_headers(lines):
    """Short lines that recur near page breaks: chapter titles repeated as headers."""
    counts = Counter(l.strip() for l in lines if l.strip())
    out = set()
    for line, n in counts.items():
        if n < 3 or len(line) > 75:
            continue
        letters = sum(c.isalpha() for c in line)
        if letters < 6:
            continue
        if line.isupper() or OCR_HEADER_RE.match(line):
            out.add(line)
    return out


def strip_page_furniture(lines, headers):
    """Remove folios and running headers sitting within 2 lines of a page break."""
    breaks = [i for i, l in enumerate(lines) if PAGE_RE.match(l.strip())]
    near = set()
    for i in breaks:
        for j in range(i - 2, i + 3):
            if 0 <= j < len(lines) and j != i:
                near.add(j)
    kept, removed = [], 0
    for i, line in enumerate(lines):
        s = line.strip()
        if i in near and s:
            if s in headers or FOLIO_RE.match(s) or (ROMAN_RE.match(s) and len(s) <= 6):
                removed += 1
                continue
        kept.append(line)
    return kept, removed


def real_hyphenated_tokens(text):
    """Hyphenated words that appear intact mid-line -> their hyphen is meaningful."""
    return {m.lower() for m in re.findall(r"\b[A-Za-z]{2,}-[A-Za-z]{2,}\b", text)}


def fix_hyphenation(lines, known):
    out, i, joined = [], 0, 0
    while i < len(lines):
        cur = lines[i]
        cur_r = cur.rstrip()
        head = re.search(r"([A-Za-z]{2,})-$", cur_r) if cur_r.endswith("-") else None
        if head and not cur_r.endswith("--"):
            # the continuation may sit past blank lines left by a lifted caption,
            # and past a page break when the word is split across pages
            j, marker = i + 1, None
            while j < len(lines) and j <= i + 3:
                s = lines[j].strip()
                if not s:
                    j += 1
                elif PAGE_RE.match(s) and marker is None:
                    marker = lines[j]
                    j += 1
                else:
                    break
            m = re.match(r"^([a-z]{2,})(\b.*)$", lines[j].strip()) if j < len(lines) else None
            if m:
                candidate = head.group(1).lower() + "-" + m.group(1).lower()
                sep = "-" if candidate in known else ""
                out.append(cur_r[:-1] + sep + m.group(1) + m.group(2))
                if marker is not None:
                    out.append(marker)
                joined += 1
                i = j + 1
                continue
        out.append(cur)
        i += 1
    return out, joined


def reflow(lines):
    """Join hard-wrapped prose lines; leave page markers, headings, equations alone."""
    out = []
    for line in lines:
        s = line.rstrip()
        if not out or not s or PAGE_RE.match(s.strip()):
            out.append(s)
            continue
        # a single blank left behind by a lifted caption is not a paragraph break
        # if the sentence clearly continues across it
        if (
            len(out) >= 2
            and not out[-1].strip()
            and out[-2].strip()
            and is_prose(out[-2])
            and not re.search(r"[.:;!?\"')]$", out[-2])
            and is_prose(s)
            and re.match(r"^[a-z]", s.strip())
        ):
            out.pop()
        prev = out[-1]
        if (
            prev
            and not PAGE_RE.match(prev.strip())
            and is_prose(prev)
            and len(prev) >= 45
            and not re.search(r"[.:;!?\"')]$", prev)
            and is_prose(s)
            and re.match(r"^[a-z(]", s.strip())
        ):
            out[-1] = prev + " " + s.strip()
        else:
            out.append(s)
    return out


CAPTION_RE = re.compile(r"^(FIGURE|TABLE|Figure|Table)\s+\d+[-.–]?\d*\b")


def lift_captions(lines):
    """
    pdftotext interleaves figure/table captions into the prose flow, often
    mid-sentence ("...cause the per-unit-" / caption / "length resistance...").
    Move each caption block to the end of its page so the sentence rejoins.
    """
    out, captions, moved = [], [], 0
    page_start = 0

    def flush():
        nonlocal captions
        if captions:
            if out and out[-1].strip():
                out.append("")
            out.extend(captions)
            captions = []

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if PAGE_RE.match(s):
            flush()
            out.append(lines[i])
            page_start = len(out)
            i += 1
            continue
        if CAPTION_RE.match(s) and len(out) > page_start:
            block = []
            while i < len(lines) and lines[i].strip() and not PAGE_RE.match(lines[i].strip()):
                block.append(lines[i])
                i += 1
            captions.extend(block)
            captions.append("")
            moved += 1
            continue
        out.append(lines[i])
        i += 1
    flush()
    return out, moved


JUNK_RE = re.compile(r"^[^\w]{1,3}$")


def drop_junk(lines):
    out, removed = [], 0
    for l in lines:
        s = l.strip()
        if s and (JUNK_RE.match(s) or set(s) <= set(". ·")):
            removed += 1
            continue
        out.append(l)
    return out, removed


def collapse_blanks(lines):
    out = []
    for l in lines:
        if not l.strip():
            if out and not out[-1].strip():
                continue
            out.append("")
        else:
            out.append(l.rstrip())
    return out


def process(name, cfg):
    raw = (SRC / name).read_text(encoding="utf-8", errors="replace")
    header = [l for l in raw.split("\n")[:3] if l.startswith("#")]
    lines = raw.split("\n")
    n_in = len(lines)

    drop = set()
    for a, b in cfg["drop_ranges"]:
        drop.update(range(a - 1, b))
    cut = cfg["truncate_at"] - 1
    lines = [l for i, l in enumerate(lines) if i < cut and i not in drop]
    after_matter = len(lines)

    lines = [normalize(l) for l in lines]
    n_banner = sum(bool(BANNER_RE.search(l)) for l in lines)
    lines = [l for l in lines if not BANNER_RE.search(l)]

    headers = find_running_headers(lines)
    lines, n_furniture = strip_page_furniture(lines, headers)

    lines, n_captions = lift_captions(lines)

    known = real_hyphenated_tokens("\n".join(lines))
    lines, n_joined = fix_hyphenation(lines, known)

    lines, n_junk = drop_junk(lines)
    lines = reflow(lines)
    lines = collapse_blanks(lines)

    body = "\n".join(lines).strip() + "\n"
    out = ("\n".join(header)
           + "\n# CLEANED: matter trimmed, banners/headers removed, reflowed\n\n"
           + body)
    (DST / name).write_text(out, encoding="utf-8")

    return {
        "book": name,
        "lines_in": n_in,
        "matter_dropped": n_in - after_matter,
        "banners": n_banner,
        "furniture": n_furniture,
        "captions_moved": n_captions,
        "hyphens_joined": n_joined,
        "junk": n_junk,
        "lines_out": len(lines),
        "words_out": len(body.split()),
        "pages": sum(1 for l in lines if PAGE_RE.match(l.strip())),
    }


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    rows = [process(n, c) for n, c in BOOKS.items()]
    w = max(len(r["book"]) for r in rows)
    cols = ["lines_in", "matter_dropped", "banners", "furniture", "captions_moved",
            "hyphens_joined", "junk", "lines_out", "words_out", "pages"]
    print("book".ljust(w) + " " + " ".join(c.rjust(14) for c in cols))
    for r in rows:
        print(r["book"].ljust(w) + " " + " ".join(str(r[c]).rjust(14) for c in cols))
