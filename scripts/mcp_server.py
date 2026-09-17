"""
MCP server exposing the signal-integrity textbook corpus to Claude Desktop.

Tools:
  search_textbooks  hybrid retrieval + cross-encoder rerank, returns cited passages
  read_context      widen a hit into its neighbouring chunks on the same page range
  list_books        what is in the corpus, for book-filtered searches

Transport is stdio by default (Claude Desktop). Set MCP_TRANSPORT=streamable-http
to serve a team over the network.

Environment:
  QDRANT_URL       shared Qdrant server; omit to use the local data/qdrant file
  QDRANT_API_KEY   if the shared server requires one
  RAG_RERANK       "0" to skip the cross-encoder (faster, noticeably worse)
"""
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# stdio transport speaks JSON-RPC on stdout -- a single stray print from a model
# loader corrupts the stream, so silence the chatty libraries up front and route
# anything that still slips through to stderr (see `quiet()` below).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _name in ("httpx", "httpcore", "huggingface_hub", "transformers", "FlagEmbedding"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from mcp.server.mcpserver import MCPServer  # noqa: E402

import search_qdrant as sq  # noqa: E402

RERANK = os.environ.get("RAG_RERANK", "1") != "0"


@contextlib.contextmanager
def quiet():
    """Send anything a tool body prints to stderr, keeping stdout for JSON-RPC."""
    with contextlib.redirect_stdout(sys.stderr):
        yield

BOOKS = {
    "paul_mtl": "Paul, Analysis of Multiconductor Transmission Lines (2e), 2008",
    "hall_asi": "Hall & Heck, Advanced Signal Integrity for High-Speed Digital Designs, 2009",
    "dsi_mod": "Digital Signal Integrity: Modeling and Simulation with Interconnects and Packages",
}

server = MCPServer(
    name="signal-integrity-books",
    version="1.0.0",
    log_level="ERROR",
    instructions=(
        "A retrieval index over three signal-integrity / high-speed digital design "
        "textbooks. Use search_textbooks to ground answers about transmission lines, "
        "crosstalk, impedance, losses, equalization, jitter, S-parameters and related "
        "topics, and cite the book, section and page returned with each passage.\n\n"
        "The passages come from PDF text extraction and OCR. Prose is reliable; "
        "equations are frequently garbled and must not be transcribed as "
        "authoritative. Describe a relationship in words and point the reader at the "
        "cited page instead."
    ),
)

_client = None


def client():
    global _client
    if _client is None:
        _client = sq.get_client()
    return _client


@server.tool(
    description=(
        "Search three signal-integrity textbooks and return the most relevant "
        "passages with book, section and page citations. Use for any question about "
        "transmission lines, crosstalk, impedance, conductor/dielectric losses, "
        "equalization, jitter, S-parameters, or PCB/package interconnect behaviour."
    )
)
def search_textbooks(query: str, k: int = 6, book: str | None = None) -> str:
    """
    Args:
        query: A natural-language question or topic. Full sentences work better
            than keywords.
        k: How many passages to return (1-15).
        book: Optional filter -- "paul_mtl", "hall_asi" or "dsi_mod". Omit to
            search all three.
    """
    k = max(1, min(int(k), 15))
    if book and book not in BOOKS:
        return f"Unknown book {book!r}. Valid values: {', '.join(BOOKS)}."

    with quiet():
        hits = sq.search(query, k=k, candidates=max(30, k * 5), book=book,
                         rerank=RERANK, client=client())
    if not hits:
        return "No passages matched that query."

    out = []
    for i, h in enumerate(hits, 1):
        out.append(
            f"[{i}] {sq.cite(h)}  (chunk_id={h['id']})\n{h['text']}"
        )
    return "\n\n---\n\n".join(out)


@server.tool(
    description=(
        "Fetch the chunks immediately before and after a chunk returned by "
        "search_textbooks, to read a derivation or argument that was cut off."
    )
)
def read_context(chunk_id: int, before: int = 1, after: int = 1) -> str:
    """
    Args:
        chunk_id: The chunk_id shown in a search_textbooks result.
        before: How many preceding chunks to include (0-3).
        after: How many following chunks to include (0-3).
    """
    from qdrant_client import models

    before = max(0, min(int(before), 3))
    after = max(0, min(int(after), 3))
    ids = list(range(int(chunk_id) - before, int(chunk_id) + after + 1))

    with quiet():
        points = client().retrieve(
            collection_name=sq.COLLECTION, ids=ids, with_payload=True
        )
    if not points:
        return f"No chunk with id {chunk_id}."

    # chunk ids are assigned in reading order, but neighbours can belong to a
    # different section -- keep only those from the same book and section
    by_id = {p.id: p.payload for p in points}
    anchor = by_id.get(int(chunk_id))
    if anchor is None:
        return f"No chunk with id {chunk_id}."

    out = []
    for i in ids:
        p = by_id.get(i)
        if not p or p["book"] != anchor["book"] or p["section"] != anchor["section"]:
            continue
        marker = " <- requested" if i == int(chunk_id) else ""
        out.append(f"[chunk {i}]{marker} {sq.cite(p)}\n{p['text']}")
    return "\n\n---\n\n".join(out)


@server.tool(description="List the books in the corpus and their coverage.")
def list_books() -> str:
    """Returns each book key, full title and indexed chunk count."""
    from qdrant_client import models

    rows = []
    for key, title in BOOKS.items():
        with quiet():
            n = client().count(
                collection_name=sq.COLLECTION,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="book", match=models.MatchValue(value=key))]
                ),
                exact=True,
            ).count
        rows.append({"book": key, "title": title, "chunks": n})
    return json.dumps(rows, indent=2)


if __name__ == "__main__":
    server.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))
