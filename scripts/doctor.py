"""
End-to-end check of the MCP server, before trusting it to a client.

    python rag.py doctor                                  # local, over stdio
    python rag.py doctor --url https://host/mcp --token X  # a deployed endpoint

Local mode runs the server as a real subprocess over stdio and fails if anything
pollutes stdout. That is the failure mode that matters for Claude Desktop: model
loaders and HTTP clients print to stdout, which corrupts the JSON-RPC stream, and
it is the usual reason a server shows up in the client with no tools. stdin is
held open for the whole session, so in-flight tool calls are not cancelled.

Remote mode checks, in order, that the endpoint is reachable, rejects an
unauthenticated request, accepts the token, advertises the tools, and returns real
passages for a live query. Both modes exit non-zero on the first hard failure, so
either can gate a deploy.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TOOLS = {"search_textbooks", "read_context", "list_books"}
TIMEOUT = 600  # the first tool call pays for the model load
QUERY = "what causes far-end crosstalk?"

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "doctor", "version": "1.0"},
    },
}


# ------------------------------------------------------------- local (stdio)

def check_stdio():
    proc = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / "mcp_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "MCP_TRANSPORT": "stdio"},
    )
    q = queue.Queue()

    def pump(pipe):
        for line in pipe:
            q.put(line)
        q.put(None)

    threading.Thread(target=pump, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=lambda: list(proc.stderr), daemon=True).start()

    ok = True
    bad_lines = []

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def await_id(want):
        while True:
            try:
                line = q.get(timeout=TIMEOUT)
            except queue.Empty:
                return None
            if line is None:
                return None
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append(line)
                continue
            if msg.get("id") == want:
                return msg

    send(INIT)
    init = await_id(1)
    if init:
        info = init.get("result", {}).get("serverInfo", {})
        print(f"  initialize  -> {info.get('name')} v{info.get('version')}")
    else:
        print("  initialize  -> MISSING")
        ok = False

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tl = await_id(2)
    if tl:
        names = {t["name"] for t in tl.get("result", {}).get("tools", [])}
        print(f"  tools/list  -> {sorted(names)}")
        if not TOOLS <= names:
            ok = False
    else:
        print("  tools/list  -> MISSING")
        ok = False

    def call(cid, name, arguments):
        nonlocal ok
        send({"jsonrpc": "2.0", "id": cid, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}})
        msg = await_id(cid)
        if not msg:
            print(f"  {name:<16} -> MISSING")
            ok = False
            return None
        result = msg.get("result", {})
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        print(f"  {name:<16} -> {len(text)} chars, isError={result.get('isError')}")
        if text:
            print(f"      {text.splitlines()[0][:96]}")
        if not text or result.get("isError"):
            ok = False
        return text

    call(3, "list_books", {})
    hit = call(4, "search_textbooks", {"query": QUERY, "k": 2})
    if hit and "chunk_id=" in hit:
        cid = int(hit.split("chunk_id=")[1].split(")")[0])
        call(5, "read_context", {"chunk_id": cid, "before": 1, "after": 1})

    proc.stdin.close()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

    if bad_lines:
        ok = False
        print(f"\n  NON-JSON ON STDOUT ({len(bad_lines)} lines) -- would break Claude Desktop:")
        for line in bad_lines[:5]:
            print(f"      {line[:110]!r}")
    else:
        print("\n  stdout was pure JSON-RPC")
    return ok


# ------------------------------------------------------------------- remote

# Cloudflare's Browser Integrity Check 403s the default urllib agent (error 1010)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def rpc(url, token, body, timeout=180, session=None):
    """
    One JSON-RPC call. The server answers as SSE, so pull the data: line out.

    streamable-http is session-oriented: initialize returns an mcp-session-id
    that every later request must carry, or the server treats the call as a new
    unbound session and refuses to serve it.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status, session = r.status, r.headers.get("mcp-session-id")
    except urllib.error.HTTPError as e:
        return e.code, None, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", None

    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                return status, json.loads(line[5:].strip()), session
            except json.JSONDecodeError:
                continue
    try:
        return status, json.loads(raw), session
    except json.JSONDecodeError:
        return status, raw[:300], session


def normalize_url(raw):
    """Repair a URL that has picked up a second scheme from being pasted twice."""
    url = raw.strip()
    while "://" in url[8:]:
        url = url[url.index("://", 5) + 3:]
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
    if url != raw.strip():
        print(f"note: cleaned up the URL -> {url}")
    if not url.rstrip("/").endswith("/mcp"):
        print("note: the endpoint path usually ends in /mcp")
    return url


def check_remote(url, token):
    url = normalize_url(url)
    print(f"checking {url}\n")
    ok = True

    print("0. endpoint is reachable")
    status, body, _ = rpc(url, token, INIT, timeout=30)
    if status is None:
        print(f"   FAIL {body}")
        print("        Could not connect at all. Check that the URL is the real one,")
        print("        that the server is running on the port the tunnel forwards to,")
        print("        and that the path ends in /mcp.\n")
        return False
    if status in (502, 503, 504):
        print(f"   FAIL HTTP {status} from the tunnel/proxy")
        print("        The tunnel is up but nothing is answering behind it. The usual")
        print("        cause is the server having started in stdio mode, where it waits")
        print("        on stdin and never binds a port. It should print a")
        print("        '[mcp] streamable-http listening' line on stderr.\n")
        return False
    print(f"   OK  responded with HTTP {status}\n")

    if token:
        print("1. unauthenticated request is rejected")
        status, _, _ = rpc(url, None, INIT)
        if status == 401:
            print("   OK  401\n")
        else:
            print(f"   FAIL got {status}, expected 401 -- the endpoint is OPEN\n")
            ok = False

    print("2. credentials are accepted")
    status, msg, session = rpc(url, token, INIT)
    if status == 200 and isinstance(msg, dict) and "result" in msg:
        info = msg["result"].get("serverInfo", {})
        print(f"   OK  {info.get('name')} v{info.get('version')}\n")
    else:
        print(f"   FAIL status={status} body={msg}\n")
        return False

    # the server expects the initialized notification before it will serve calls
    rpc(url, token, {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session=session)

    print("3. tools are advertised")
    _, msg, _ = rpc(url, token, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    session=session)
    names = {t["name"] for t in msg["result"].get("tools", [])} \
        if isinstance(msg, dict) and "result" in msg else set()
    if TOOLS <= names:
        print(f"   OK  {sorted(names)}\n")
    else:
        print(f"   FAIL got {sorted(names)}, expected {sorted(TOOLS)}\n")
        ok = False

    print("4. a real search returns passages (first call loads models, be patient)")
    _, msg, _ = rpc(url, token, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "search_textbooks", "arguments": {"query": QUERY, "k": 2}},
    }, timeout=TIMEOUT, session=session)
    text = ""
    if isinstance(msg, dict) and "result" in msg and not msg["result"].get("isError"):
        content = msg["result"].get("content", [])
        text = content[0].get("text", "") if content else ""
    if text and "|" in text:
        print(f"   OK  {len(text)} chars")
        print(f"       {text.splitlines()[0][:100]}\n")
    else:
        print(f"   FAIL no passages: {str(msg)[:300]}\n")
        ok = False
    return ok


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="deployed endpoint, including /mcp")
    ap.add_argument("--token", help="bearer token for --url")
    args = ap.parse_args()

    ok = check_remote(args.url, args.token) if args.url else check_stdio()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
