"""
Stage 2: extracted/<key>.txt -> data/clean/<key>.txt, ready for chunking.

Per document:
  1. drop front matter (table of contents) and back matter (index), by page
  2. strip per-page publisher banners (corpus.json: banner_patterns)
  3. strip running headers / footers and bare folio numbers around [PAGE n]
  4. normalize unicode (ligatures, smart quotes, nbsp, soft hyphen)
  5. repair line-break hyphenation, using corpus evidence to decide whether the
     hyphen is real ("high-speed") or a soft wrap hyphen ("propaga-tion")
  6. reflow hard-wrapped prose into paragraphs, leaving equation lines alone
  7. drop pure-punctuation junk lines and collapse blank runs

Nothing here is per-book. The only document-specific inputs are `skip_pages` and
`body_end_page` in corpus.json, expressed as page numbers so they survive a
re-extraction with a different engine. Both are auto-detected for a document that
has not set them, and the guess is written back for you to correct.

[PAGE n] markers are preserved: the chunker uses them for page citations and
strips them before embedding.
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

PAGE_RE = corpus.PAGE_RE
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


def trim_matter(lines, skip_pages, end_page):
    """Drop whole pages: the contents block(s), and everything from the index on."""
    skip = set()
    for a, b in skip_pages or []:
        skip.update(range(int(a), int(b) + 1))
    out, page, dropped = [], None, 0
    for line in lines:
        m = PAGE_RE.match(line.strip())
        if m:
            page = int(m.group(1))
            if end_page is not None and page >= int(end_page):
                dropped += 1
                break
        if page in skip:
            dropped += 1
            continue
        out.append(line)
    return out, dropped


def resolve_matter(key, doc, text, redetect=False):
    """Fill in skip_pages / body_end_page for a document that has not set them."""
    auto = set(doc.get("auto") or [])
    want_front = "skip_pages" in auto and (redetect or not doc.get("skip_pages"))
    want_end = "body_end_page" in auto and (redetect or doc.get("body_end_page") is None)
    if not (want_front or want_end):
        return False
    front, end = corpus.detect_matter(text)
    if want_front:
        doc["skip_pages"] = front
    if want_end:
        doc["body_end_page"] = end
    print(f"  {key}: detected contents {doc.get('skip_pages')}, "
          f"index starts p{doc.get('body_end_page')} "
          f"-- correct these in corpus.json if wrong")
    return True


def process(key, doc, banner_re):
    src = corpus.doc_paths(key, doc)["text"]
    raw = src.read_text(encoding="utf-8", errors="replace")
    header = [l for l in raw.split("\n")[:3] if l.startswith("#")]
    lines = raw.split("\n")
    n_in = len(lines)

    lines, n_matter = trim_matter(lines, doc.get("skip_pages"), doc.get("body_end_page"))

    lines = [normalize(l) for l in lines]
    n_banner = 0
    if banner_re is not None:
        n_banner = sum(bool(banner_re.search(l)) for l in lines)
        lines = [l for l in lines if not banner_re.search(l)]

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
    corpus.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    corpus.doc_paths(key, doc)["clean"].write_text(out, encoding="utf-8")

    return {
        "book": key,
        "lines_in": n_in,
        "matter_dropped": n_matter,
        "banners": n_banner,
        "furniture": n_furniture,
        "captions_moved": n_captions,
        "hyphens_joined": n_joined,
        "junk": n_junk,
        "lines_out": len(lines),
        "words_out": len(body.split()),
        "pages": sum(1 for l in lines if PAGE_RE.match(l.strip())),
    }


def clean_signature(doc):
    """What a re-clean depends on: the extracted text and the trim settings."""
    return "|".join(str(doc.get(k)) for k in ("text_hash", "skip_pages", "body_end_page"))


def run(keys=None, force=False, redetect=False, quiet=False):
    """Clean every document whose text or trim settings changed. Returns stat rows."""
    cfg = corpus.load()
    banners = [p for p in cfg.get("banner_patterns") or [] if p]
    banner_re = re.compile("|".join(f"(?:{p})" for p in banners)) if banners else None

    rows, skipped, dirty = [], [], False
    for key, doc in cfg["documents"].items():
        if keys and key not in keys:
            continue
        paths = corpus.doc_paths(key, doc)
        if not paths["text"].exists():
            continue
        text = paths["text"].read_text(encoding="utf-8", errors="replace")
        dirty |= resolve_matter(key, doc, text, redetect)
        digest = corpus.file_hash(paths["text"])
        if doc.get("text_hash") != digest:
            doc["text_hash"] = digest
            dirty = True

        sig = clean_signature(doc)
        if not force and paths["clean"].exists() and doc.get("clean_sig") == sig:
            skipped.append(key)
            continue
        if not quiet:
            print(f"  cleaning {key}", flush=True)
        rows.append(process(key, doc, banner_re))
        doc["clean_sig"] = sig
        dirty = True

    if dirty:
        corpus.save(cfg)
    return cfg, rows, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="document keys to clean")
    ap.add_argument("--force", action="store_true", help="re-clean even if unchanged")
    ap.add_argument("--redetect", action="store_true",
                    help="re-run front/back matter detection, overwriting auto values")
    args = ap.parse_args()

    _, rows, skipped = run(args.only, args.force, args.redetect)
    if skipped:
        print(f"unchanged: {', '.join(skipped)}")
    if not rows:
        return
    w = max(len(r["book"]) for r in rows)
    cols = ["lines_in", "matter_dropped", "banners", "furniture", "captions_moved",
            "hyphens_joined", "junk", "lines_out", "words_out", "pages"]
    print("book".ljust(w) + " " + " ".join(c.rjust(14) for c in cols))
    for r in rows:
        print(r["book"].ljust(w) + " " + " ".join(str(r[c]).rjust(14) for c in cols))


if __name__ == "__main__":
    main()
