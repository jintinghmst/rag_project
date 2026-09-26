"""
Stage 0: walk original/ and register every document in corpus.json.

Two kinds of PDF live side by side in this corpus and they cannot be treated
alike:

  * a **book** or a standalone **article** -- one document, one PDF.
  * a whole journal **issue** -- one PDF holding ~20 unrelated articles. Indexed
    as a single document it would cite "Volume 70, Issue 3" instead of a paper,
    and read_context would happily walk from the end of one paper into the start
    of another. So an issue is split: each article becomes its own document with
    its own title, its own page range, and its own chunk-id block.

Articles are found from the PDF's bookmarks, which IEEE issue files carry one per
article. Titles come from the first page of each article -- the largest type on
an IEEE first page is the title -- because the bookmark labels are often just
sequence numbers ("01", "02") or the publisher's internal filenames.

    python scripts/scan_pdfs.py            # register anything new
    python scripts/scan_pdfs.py --rescan   # re-read every PDF's structure
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

# An issue is a big PDF with many bookmarks. Both conditions matter: a book can
# have many bookmarks (its chapters) but a book's bookmarks are not articles, and
# a long PDF with three bookmarks is a book with a sparse outline.
MIN_ARTICLE_BOOKMARKS = 6
MIN_ISSUE_PAGES = 80
MIN_ARTICLE_PAGES = 2

# Front matter inside an issue: covers, contents, editorials, indexes. None of it
# is a paper, and the contents pages in particular rank highly for any query.
SKIP_TITLE_RE = re.compile(
    r"^\s*(table of )?contents\b|^\s*cover\b|^\s*front cover|^\s*back cover|"
    r"^\s*(guest )?editorial\b|^\s*information for authors|^\s*call for papers|"
    r"^\s*(z\s+)?regular papers\s*$|^\s*special issue papers\s*$|"
    # a running header captured instead of a title, e.g.
    # "3062 IEEE TRANSACTIONS ON MICROWAVE THEORY AND TECHNIQUES, VOL. 73, ..."
    r"^\s*\d{0,5}\s*ieee transactions on\b.*\bvol\.|"
    r"^\s*instructions? for authors|^\s*\d{4} index\b|^\s*index\b|"
    r"^\s*ieee (transactions|xplore)\b.*\bcontents\b|^\s*blank page",
    re.I)
SKIP_BOOKMARK_RE = re.compile(
    r"contents|cover|editorial|adverti|masthead|call[-_ ]?for|index|blank", re.I)

# a standalone "2022 Index IEEE Transactions on ... Vol. 1.pdf"
INDEX_FILE_RE = re.compile(r"\bindex\b.*\bvol", re.I)


def open_pdf(path):
    import pymupdf

    return pymupdf.open(path)


def first_level_bookmarks(toc):
    """[(title, start_page)] for the outermost outline level, in page order."""
    if not toc:
        return []
    top = min(level for level, _, _ in toc)
    out = []
    for level, title, page in toc:
        if level == top and page and page > 0:
            out.append((title.strip(), page))
    out.sort(key=lambda t: t[1])
    return out


def article_spans(doc):
    """[(bookmark_title, first_page, last_page)] 1-based inclusive, for an issue."""
    marks = first_level_bookmarks(doc.get_toc())
    spans = []
    for i, (title, start) in enumerate(marks):
        end = (marks[i + 1][1] - 1) if i + 1 < len(marks) else doc.page_count
        if end >= start:
            spans.append((title, start, end))
    return spans


def looks_like_issue(doc):
    return (doc.page_count >= MIN_ISSUE_PAGES
            and len(first_level_bookmarks(doc.get_toc())) >= MIN_ARTICLE_BOOKMARKS)


def page_title(page):
    """
    The largest run of text on the page, which on an IEEE first page is the title.

    Spans are grouped by rounded font size; the biggest size that yields something
    title-shaped wins. Falls back to the longest early line.
    """
    try:
        data = page.get_text("dict")
    except Exception:
        return ""
    by_size = {}
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                size = round(span.get("size", 0), 1)
                x, y = span.get("origin", (0, 0))
                # reading order: down the page, then across. Rounding y to whole
                # points groups a line's spans together, so words on one line are
                # not interleaved with the line below when the baseline wobbles.
                by_size.setdefault(size, []).append((round(y), round(x), text))

    for size in sorted(by_size, reverse=True):
        parts = [t for _, _, t in sorted(by_size[size])]
        title = re.sub(r"\s+", " ", " ".join(parts)).strip()
        # a title is a phrase: long enough to be one, not a page of body text,
        # and not the journal's own running header
        if 12 <= len(title) <= 300 and not re.match(r"^ieee\b", title, re.I):
            letters = sum(c.isalpha() for c in title)
            if letters >= 10:
                return title
    return ""


PRINTED_PAGE_RE = re.compile(r"\b(\d{1,5})\b")


def _page_number_candidates(page):
    """Every integer in the page's header and footer -- the folio is among them."""
    try:
        lines = [l.strip() for l in page.get_text("text").split("\n") if l.strip()]
    except Exception:
        return set()
    out = set()
    for line in lines[:4] + lines[-3:]:
        for match in PRINTED_PAGE_RE.finditer(line):
            value = int(match.group(1))
            if 1 <= value <= 30000:
                out.add(value)
    return out


def issue_page_offset(doc, spans, sample=8):
    """
    One offset for the whole issue: printed folio - pdf index.

    Derived by agreement rather than by parsing, because the header line also
    carries the volume, the issue number and the year, and any of those can look
    like a page number on a single page. Only the real folio advances in step
    with the PDF, so only it yields the same offset on every article sampled --
    the volume and year produce a different offset each time. Fewer than three
    articles agreeing means no confident answer, and citations fall back to the
    PDF's own paging.
    """
    from collections import Counter

    votes = Counter()
    for _, start, _ in spans[:sample]:
        for value in _page_number_candidates(doc[start - 1]):
            offset = value - start
            # Usually negative: an issue's unpaginated covers and contents push
            # the printed folio behind the PDF index. Bounded so a stray year or
            # ISSN fragment cannot win, however consistently it appears.
            if -400 <= offset <= 30000:
                votes[offset] += 1
    if not votes:
        return None
    offset, agreement = votes.most_common(1)[0]
    return offset if agreement >= 3 else None


def source_label(rel_path):
    """'IEEE Transactions on ... / 2024' from the folder the PDF was filed under."""
    parts = Path(rel_path).parts[:-1]
    return " / ".join(parts) if parts else ""


def unique_key(cfg, base):
    key = base or "doc"
    if key not in cfg["documents"]:
        return key
    n = 2
    while f"{key}_{n}" in cfg["documents"]:
        n += 1
    return f"{key}_{n}"


def article_key(cfg, rel_path, index, title):
    stem = corpus.slug(Path(rel_path).stem)[:28]
    hint = corpus.slug(title)[:24] if title else ""
    return unique_key(cfg, f"{stem}_{index:02d}" + (f"_{hint}" if hint else ""))


def register_pdf(cfg, path, rescan=False):
    """Add this PDF -- as one document, or as one document per article."""
    rel = corpus.rel_pdf(path)
    known = {d.get("pdf") for d in cfg["documents"].values()}
    if rel in known and not rescan:
        return []

    # The same PDF filed somewhere new -- a book moved into Books/, a paper
    # refiled by year. Match on basename and follow the move, otherwise the
    # document is registered twice and its text is indexed twice.
    name = Path(rel).name
    moved = [k for k, d in cfg["documents"].items()
             if d.get("pdf") and Path(d["pdf"]).name == name]
    if moved:
        for key in moved:
            cfg["documents"][key]["pdf"] = rel
        return []

    if INDEX_FILE_RE.search(path.name):
        return []   # a volume index, not a paper

    added = []
    with open_pdf(path) as doc:
        if not looks_like_issue(doc):
            title = page_title(doc[0]) if doc.page_count else ""
            key = unique_key(cfg, corpus.short_key(path.stem))
            cfg["documents"][key] = {
                "pdf": rel,
                "title": title or re.sub(r"\s+", " ", path.stem).strip(),
                "kind": "article" if doc.page_count < MIN_ISSUE_PAGES else "book",
                "source": source_label(rel),
                "slot": corpus.next_slot(cfg),
                "text": f"{key}.txt",
                "skip_pages": [],
                "body_end_page": None,
                "auto": ["title", "skip_pages", "body_end_page"],
            }
            added.append(key)
            return added

        spans = article_spans(doc)
        offset = issue_page_offset(doc, spans)
        for i, (bookmark, start, end) in enumerate(spans, 1):
            if end - start + 1 < MIN_ARTICLE_PAGES:
                continue
            if SKIP_BOOKMARK_RE.search(bookmark):
                continue
            title = page_title(doc[start - 1])
            if not title or SKIP_TITLE_RE.search(title):
                continue
            key = article_key(cfg, rel, i, title)
            cfg["documents"][key] = {
                "pdf": rel,
                "title": title,
                "kind": "article",
                "source": source_label(rel),
                "pages": [start, end],
                "page_offset": offset,
                "slot": corpus.next_slot(cfg),
                "text": f"{key}.txt",
                "skip_pages": [],
                "body_end_page": None,
                # an article has no contents block and no index to detect
                "auto": [],
            }
            added.append(key)
    return added


def shouty(title):
    """
    Mostly-capitalised: a section divider or a mangled small-caps heading.

    IEEE sets some front matter in small caps with an oversized initial, which
    reads back as "INKED PECIAL SSUES" once the initials are grouped separately.
    Real paper titles are title case, so this catches the wreckage without
    risking a genuine one.
    """
    words = [w for w in title.split() if len(w) > 2]
    if not words:
        return True
    return sum(1 for w in words if w.isupper()) / len(words) > 0.6


def prune(cfg, quiet=False):
    """Drop registered documents that turned out not to be papers."""
    doomed = [k for k, d in cfg["documents"].items()
              if d.get("kind") == "article"
              and (SKIP_TITLE_RE.search(d["title"]) or shouty(d["title"]))]
    for key in doomed:
        paths = corpus.doc_paths(key, cfg["documents"][key])
        for name in ("text", "clean", "chunks"):
            if paths[name]:
                paths[name].unlink(missing_ok=True)
        if not quiet:
            print(f"  pruned {key}: {cfg['documents'][key]['title'][:60]}")
        del cfg["documents"][key]
    return doomed


def run(rescan=False, limit=None, quiet=False):
    cfg = corpus.load()
    files = corpus.pdfs()
    if limit:
        files = files[:limit]
    added, failed = [], []
    for n, path in enumerate(files, 1):
        try:
            new = register_pdf(cfg, path, rescan)
        except Exception as exc:
            failed.append((corpus.rel_pdf(path), f"{type(exc).__name__}: {exc}"))
            continue
        added += new
        if new and not quiet:
            print(f"  [{n}/{len(files)}] {Path(path).name[:58]} -> {len(new)} document(s)",
                  flush=True)
    removed = prune(cfg, quiet)
    corpus.save(cfg)
    return cfg, added, failed, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rescan", action="store_true",
                    help="re-read PDFs already in the registry")
    ap.add_argument("--limit", type=int, help="only the first N PDFs (a dry run)")
    args = ap.parse_args()

    cfg, added, failed, removed = run(args.rescan, args.limit)
    for rel, err in failed:
        print(f"! {rel}: {err}")
    kinds = {}
    for doc in cfg["documents"].values():
        kinds[doc.get("kind", "book")] = kinds.get(doc.get("kind", "book"), 0) + 1
    print(f"\nregistered {len(added)} new, pruned {len(removed)}; corpus now holds "
          f"{len(cfg['documents'])} ({kinds})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
