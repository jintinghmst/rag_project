# Signal-integrity RAG corpus

Vector store over three signal-integrity textbooks, built for a RAG chatbot.

- **Embedding model:** `BAAI/bge-m3` — 8192-token context, and one forward pass
  yields both a dense vector and a learned sparse (lexical) vector. The sparse
  half is what makes queries hinging on notation (`Z0`, `S21`, `ε_r`,
  "equation 4.48") work; dense vectors alone handle those poorly.
- **Reranker:** `BAAI/bge-reranker-v2-m3` over the fused top-30. All three books
  discuss crosstalk, impedance and loss in near-identical language, so the
  cross-encoder is what separates the right chapter from a plausible one.
- **Store:** Qdrant in local (embedded) mode at `data/qdrant`, one collection
  `signal_integrity` with named vectors `dense` + `sparse`, fused with RRF.

## Setup

    python -m venv .venv
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    .venv\Scripts\python -m pip install -r requirements.txt

## Pipeline

    .venv\Scripts\python scripts\clean_corpus.py    # training/ -> data/clean/
    .venv\Scripts\python scripts\chunk_corpus.py    # -> data/chunks.jsonl
    .venv\Scripts\python scripts\ingest_qdrant.py --recreate
    .venv\Scripts\python scripts\search_qdrant.py "why is far-end crosstalk zero in a homogeneous medium?"

`search_qdrant.search()` is importable and returns hits with full metadata —
that is the function to wire into anything else.

## Use it from Claude Desktop (MCP)

`scripts/mcp_server.py` exposes the corpus as three tools:

| tool | what it does |
|---|---|
| `search_textbooks(query, k, book)` | hybrid retrieval + rerank, returns cited passages |
| `read_context(chunk_id, before, after)` | widens a hit into neighbouring chunks of the same section |
| `list_books()` | the three books and their chunk counts |

Merge `claude_desktop_config.example.json` into your Claude Desktop config and
restart the app. On Windows that file lives at
`%APPDATA%\Claude\claude_desktop_config.json`; on macOS,
`~/Library/Application Support/Claude/claude_desktop_config.json`.

Paths in the config must be absolute — Claude Desktop does not run the server
with this repo as its working directory. Point `command` at the venv's
`python.exe`, not a bare `python`.

Verify the server independently of Claude Desktop with:

    .venv\Scripts\python scripts\test_mcp_stdio.py

That drives a real subprocess over stdio and fails if anything pollutes stdout,
which is the usual reason an MCP server shows up empty in the client. The first
tool call loads ~2.8 GB of models and takes a minute; later calls are ~1-2 s.

## Sharing with a team

`data/qdrant` is an embedded single-file store behind a lock — one process at a
time. It cannot back several teammates at once, and copying the file around means
everyone drifts. Run one Qdrant instead:

    cp .env.example .env          # set QDRANT_API_KEY
    docker compose up -d
    QDRANT_URL=http://<host>:6333 QDRANT_API_KEY=... \
        .venv\Scripts\python scripts\ingest_qdrant.py --recreate

Each teammate then clones the repo, installs the requirements, and sets
`QDRANT_URL` / `QDRANT_API_KEY` in the `env` block of their Claude Desktop
config. `scripts/search_qdrant.get_client()` picks the shared server up
automatically; with the vars unset it falls back to the local file, so a solo
setup needs no changes.

What each teammate still needs locally: the Python env and ~2.8 GB of model
weights (bge-m3 + the reranker), downloaded on first use to their Hugging Face
cache. Only the index is centralized — embedding a query still happens on their
machine. It runs on CPU at roughly 1-3 s per query without a GPU.

Qdrant has no user accounts; `QDRANT_API_KEY` is the only access control, so keep
the container on a VPN or campus network rather than the public internet.

Note that the index contains substantial verbatim text from three copyrighted
textbooks — including one retrieved through a university subscription. Sharing it
beyond people already licensed for those books is a redistribution question worth
settling before you hand out the URL.

## Contents

| book key | source | pages | chunks |
|---|---|---|---|
| `paul_mtl` | Paul, *Analysis of Multiconductor Transmission Lines*, 2e | 779 | 1043 |
| `hall_asi` | Hall & Heck, *Advanced Signal Integrity for High-Speed Digital Designs* | 652 | 902 |
| `dsi_mod`  | *Digital Signal Integrity: Modeling and Simulation* (scanned, OCR) | 526 | 552 |

2497 chunks, p50 433 tokens. Each chunk carries `book`, `chapter`, `section`,
`section_title`, `page_start`, `page_end` — enough to cite a source in an answer,
and indexed in Qdrant so retrieval can be filtered by book or chapter.

## Cleanup notes

`clean_corpus.py` drops each book's table of contents and index, strips the
per-page Wiley download banner (651 in Hall) and running headers/folios,
normalizes unicode, repairs hyphenation split across line and page breaks, and
reflows hard-wrapped prose into paragraphs.

The load-bearing step is caption relocation: `pdftotext` drops figure and table
captions into the middle of sentences ("...cause the per-unit-" / *FIGURE 3.3 ...*
/ "length resistance matrix..."). ~1180 caption blocks are moved to the end of
their page so the sentence rejoins.

Line ranges for front/back matter in `BOOKS` are hard-coded per file and were
verified by inspection — re-check them if the extractions are regenerated.

## Known limitations

- Equations are garbled in all three extractions (inline text, not LaTeX) and
  badly garbled in the OCR'd book. Retrieval works off surrounding prose; the
  chatbot should not be trusted to reproduce a formula verbatim from a chunk.
- 26 chunks have no section assigned (chapter front pages, appendix material).
- One 1408-token chunk (a numeric table in Hall S14.10) is truncated at the
  1024-token encode limit.
