# Signal-integrity RAG

A hybrid-retrieval vector index over a shelf of signal-integrity textbooks,
served to Claude as an MCP connector, plus a terminal chatbot over the same
index.

Adding a book is dropping a PDF into `original/` and running the build again.
Nothing in the pipeline is per-book: extraction, front/back-matter trimming,
cleaning, chunking and ingest are all driven by `corpus.json`, which registers new
PDFs by itself.

## Run it

Two ways, both one command. Pick Docker if you want other people to use it;
pick native if you want a GPU, the chatbot, or Claude Desktop over stdio.

**Docker — the whole system, ready for teammates:**

```bash
./deploy.sh              # Linux/macOS      .\deploy.ps1     on Windows
```

Generates `.env` with fresh secrets, starts Qdrant, builds the index from
everything in `original/`, and serves the MCP endpoint. Re-run it after adding a
PDF; it re-embeds only what changed. Add `--tls` / `-Tls` to run Caddy in front
for a public HTTPS domain (set `DOMAIN` in `.env` first).

**Native — on this machine:**

```bash
python rag.py up         # venv + dependencies + index + server
```

or one stage at a time:

```bash
python rag.py setup                       # .venv, torch matched to your GPU, deps
python rag.py build                       # original/*.pdf -> vector index
python rag.py status                      # what is registered, built and indexed
python rag.py remove <key>                # drop a document from the index
python rag.py search "why is far-end crosstalk zero in a homogeneous medium?"
python rag.py chat                        # RAG chatbot (needs ANTHROPIC_API_KEY)
python rag.py serve --stdio               # MCP for Claude Desktop
python rag.py serve --tunnel cloudflare   # public HTTPS URL for Claude's connector UI
python rag.py doctor                      # end-to-end self test
```

`rag.py` re-executes itself inside `.venv`, so there is no activate step and no
way to run half the pipeline against the wrong interpreter.

## Adding a document

1. Put the PDF in `original/`.
2. `./deploy.sh`, or `python rag.py build`.

The build registers it in `corpus.json` under a key derived from the filename,
extracts its text, guesses where the table of contents and the index are, chunks
it, embeds it, and adds it to the collection. Every other book is skipped — each
stage is cached on a content hash, and the ingest compares against a manifest
stored beside the index, so it stays correct against a shared Qdrant that this
machine did not build.

Then open `corpus.json` and fix what the guesses got wrong:

```jsonc
"hall_asi": {
  "pdf": "Advanced Signal Integrity ... Hall.pdf",
  "title": "Hall & Heck, Advanced Signal Integrity for High-Speed Digital Designs",
  "slot": 1,                     // owns chunk ids [slot*2^20, (slot+1)*2^20)
  "skip_pages": [[4, 12]],       // the table of contents, by PDF page
  "body_end_page": 662,          // the index starts here; everything after is dropped
  "auto": []                     // fields still open to re-guessing; empty = hand-set
}
```

`skip_pages` and `body_end_page` matter more than they look: a contents page is a
dense list of every term in the book and outranks real passages for almost any
query. Check them on a new book — `python rag.py build --only <key> --force` after
an edit. Removing a key from `auto` pins your value.

A scanned PDF with no text layer is reported rather than indexed empty; OCR it
first (`ocrmypdf in.pdf out.pdf`) and build again.

Deleting a PDF does not remove it from the index — `python rag.py remove <key>`
does, and `--purge` deletes its files too.

## How it works

| stage | script | output |
|---|---|---|
| 1. extract | `extract_pdf.py` | `extracted/<key>.txt`, one `[PAGE n]` marker per page |
| 2. clean | `clean_corpus.py` | `data/clean/<key>.txt` |
| 3. chunk | `chunk_corpus.py` | `data/chunks/<key>.jsonl` |
| 4. ingest | `ingest_qdrant.py` | Qdrant collection `signal_integrity` |
| query | `search_qdrant.py` | `search()` returns hits with full metadata |

- **Embedding:** `BAAI/bge-m3` — 8192-token context, and one forward pass yields
  both a dense vector and a learned sparse (lexical) vector. The sparse half is
  what makes queries hinging on notation (`Z0`, `S21`, `ε_r`, "equation 4.48")
  work; dense vectors alone handle those poorly.
- **Reranker:** `BAAI/bge-reranker-v2-m3` over the fused top-30. The books all
  discuss crosstalk, impedance and loss in near-identical language, so the
  cross-encoder is what separates the right chapter from a plausible one.
- **Store:** Qdrant — an embedded file at `data/qdrant` by default, or a server if
  `QDRANT_URL` is set. One collection with named vectors `dense` + `sparse`, fused
  with RRF, and a `signal_integrity__manifest` collection recording what was
  loaded from where.

Chunks never cross a section boundary; within a section, paragraphs are packed to
~450 tokens with overlap. Each carries `book`, `chapter`, `section`,
`section_title`, `page_start`, `page_end` — enough to cite a source, and indexed
so retrieval can be filtered by book or chapter.

Chunk ids are allocated from a per-document block (`slot`), so adding a book never
renumbers another — which is what lets `read_context` walk to neighbouring
passages by id, and lets the ingest replace one book without touching the rest.

### Cleaning

`clean_corpus.py` drops the table of contents and the index, strips per-page
publisher banners (`banner_patterns` in `corpus.json`) and running
headers/folios, normalizes unicode, repairs hyphenation split across line and
page breaks, and reflows hard-wrapped prose into paragraphs.

The load-bearing step is caption relocation: PDF text extraction drops figure and
table captions into the middle of sentences ("...cause the per-unit-" / *FIGURE
3.3 ...* / "length resistance matrix..."). Caption blocks are moved to the end of
their page so the sentence rejoins — ~1180 of them across the current three books.

## Use it from Claude

Three tools, in both stdio and remote modes:

| tool | what it does |
|---|---|
| `search_textbooks(query, k, book)` | hybrid retrieval + rerank, returns cited passages |
| `read_context(chunk_id, before, after)` | widens a hit into neighbouring chunks of the same section |
| `list_books()` | what is indexed, with chunk counts |

**Claude Desktop (local, stdio):** merge `claude_desktop_config.example.json` into
your Claude Desktop config, with the paths made absolute, and restart the app.
Verify it independently with `python rag.py doctor`, which drives a real
subprocess over stdio and fails if anything pollutes stdout — the usual reason an
MCP server shows up empty in the client. The first tool call loads ~2.8 GB of
models and takes a minute; later calls are ~1–2 s.

**Claude connector (remote):** `./deploy.sh`, or `python rag.py serve --tunnel
cloudflare` for a throwaway URL. Add `<MCP_PUBLIC_URL>/mcp` as a custom connector.
Claude's connector UI performs OAuth and will not accept a bare bearer token, so
the server runs a minimal authorization server gated on one team passphrase
(`MCP_TEAM_PASSWORD`). Everyone who knows it can connect; there is no per-user
identity and no way to revoke one person without rotating for everyone. For
either, put a real IdP in front.

`MCP_AUTH_TOKEN` enables static bearer auth instead, for scripted clients:
`python rag.py doctor --url https://host/mcp --token <token>`.

## Sharing the index

`data/qdrant` is an embedded single-file store behind a lock — one process at a
time. It cannot back several teammates at once, and copying the file around means
everyone drifts. The Docker stack runs a real Qdrant instead; against it, one
person builds and everyone else just queries.

To point the local scripts at a shared server, set `QDRANT_URL` and
`QDRANT_API_KEY` (in `.env`, or in the `env` block of a Claude Desktop config).
With them unset everything falls back to the local file, so a solo setup needs no
configuration.

What each teammate still needs, in native mode: the Python env and ~2.8 GB of
model weights, downloaded on first use to their Hugging Face cache. Only the index
is centralized — embedding a query still happens on their machine, ~1–3 s on CPU.
The Docker deployment moves that to the server too, and teammates need nothing.

Qdrant has no user accounts; `QDRANT_API_KEY` is the only access control, so keep
it on a VPN or campus network rather than the public internet.

## Contents

| book key | source | pages | chunks |
|---|---|---|---|
| `paul_mtl` | Paul, *Analysis of Multiconductor Transmission Lines*, 2e | 779 | 1043 |
| `hall_asi` | Hall & Heck, *Advanced Signal Integrity for High-Speed Digital Designs* | 652 | 902 |
| `dsi_mod` | *Digital Signal Integrity: Modeling and Simulation* (scanned, OCR) | 515 | 530 |

2475 chunks, p50 434 tokens. `python rag.py status` prints the current state.

## Known limitations

- Equations are garbled in all three extractions (inline text, not LaTeX) and
  badly garbled in the OCR'd book. Retrieval works off surrounding prose; the
  chatbot should not be trusted to reproduce a formula verbatim from a chunk. The
  MCP server tells the model this, and the chat system prompt does too.
- `extracted/dsi_mod.txt` is OCR output for a scanned book. It cannot be
  regenerated from its PDF, which is why `extracted/` is committed rather than
  treated as a build artefact.
- Front/back-matter detection is a heuristic. It reproduces the hand-verified
  ranges on the three books here, but check `corpus.json` after adding a fourth.
- The Docker stack has not been run: Docker is not installed on the machine this
  was written on. The native pipeline, both MCP transports, auth and the
  self-tests are verified. Expect to debug the image build once, on the host.

## Licensing

The index holds substantial verbatim text from copyrighted textbooks, including
one retrieved through a university subscription. A personal local index is one
thing; an org-wide connector puts that text in front of everyone in the
organization, which is a different act. Worth settling with your library's
licensing contact before handing out the URL.
