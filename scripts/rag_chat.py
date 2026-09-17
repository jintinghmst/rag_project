"""
RAG chatbot over the signal-integrity textbooks.

    python scripts/rag_chat.py                          # interactive
    python scripts/rag_chat.py "what sets Z0 of a stripline?"   # one-shot
    python scripts/rag_chat.py --book hall_asi -k 8

Each turn: rewrite the question to stand alone (follow-ups like "and for
stripline?" retrieve nothing on their own), retrieve with hybrid search +
cross-encoder rerank, then answer from the retrieved passages only, with
[n] citations back to book / section / page.

Needs credentials: set ANTHROPIC_API_KEY, or run `ant auth login`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_qdrant import search, cite  # noqa: E402

import anthropic  # noqa: E402

MODEL = "claude-opus-5"
REWRITE_MODEL = "claude-haiku-4-5"

SYSTEM = """You are a research assistant for a PhD student working on signal \
integrity and high-speed digital design. You answer strictly from the textbook \
passages supplied in each turn.

Rules:
- Ground every claim in the passages. Cite them inline as [1], [2], matching the \
numbered sources.
- If the passages do not answer the question, say so plainly and say what they do \
cover. Do not fill the gap from background knowledge without flagging it as outside \
the sources.
- The passages come from PDF text extraction and OCR, so equations are frequently \
garbled. Never transcribe a formula as if it were authoritative: state the \
relationship in words, and point the student at the cited page to read the real \
equation. If a passage's math is unreadable, say that.
- Where the three books differ in notation or approach, say which book you are \
following.
- Be direct and technical. The reader knows electromagnetics."""


def rewrite_query(client, history, question):
    """Condense a follow-up into a standalone query for retrieval."""
    if not history:
        return question
    transcript = "\n".join(
        f"{m['role']}: {m['content'] if isinstance(m['content'], str) else ''}"
        for m in history[-6:]
    )
    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=200,
        system=(
            "Rewrite the user's latest message as a standalone search query for a "
            "signal-integrity textbook corpus. Resolve pronouns and elisions from the "
            "conversation. Output only the query, no preamble."
        ),
        messages=[{"role": "user", "content": f"{transcript}\n\nLatest: {question}"}],
    )
    out = next((b.text for b in resp.content if b.type == "text"), "").strip()
    return out or question


def build_context(hits):
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] {cite(h)}\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def answer(client, history, question, hits, stream=True):
    context = build_context(hits)
    user_block = (
        f"<passages>\n{context}\n</passages>\n\n"
        f"Question: {question}"
    )
    messages = history + [{"role": "user", "content": user_block}]

    if not stream:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        print(text)
        return text

    text = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=messages,
    ) as s:
        for event in s.text_stream:
            print(event, end="", flush=True)
            text.append(event)
    print()
    return "".join(text)


def turn(client, history, question, k, book, rerank=True, stream=True):
    query = rewrite_query(client, history, question)
    if query != question:
        print(f"  (searching: {query})\n", file=sys.stderr)

    hits = search(query, k=k, book=book, rerank=rerank)
    if not hits:
        print("No passages retrieved.")
        return

    print("Sources:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {cite(h)}")
    print()

    reply = answer(client, history, question, hits, stream=stream)

    # keep the plain question in history, not the stuffed passages, so the
    # context does not grow by k chunks every turn
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})


def main():
    # the corpus is full of math symbols; a cp936/cp1252 console raises
    # UnicodeEncodeError mid-print and truncates the answer
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="ask once and exit")
    ap.add_argument("-k", type=int, default=6, help="passages to send to Claude")
    ap.add_argument("--book", default=None, help="paul_mtl | hall_asi | dsi_mod")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    history = []

    if args.question:
        turn(client, history, args.question, args.k, args.book, args.rerank, args.stream)
        return

    print("Signal-integrity RAG chat. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q.lower() in {"exit", "quit"}:
            return
        if not q:
            continue
        print()
        turn(client, history, q, args.k, args.book, args.rerank, args.stream)
        print()


if __name__ == "__main__":
    main()
