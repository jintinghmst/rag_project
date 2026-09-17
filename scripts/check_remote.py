"""
Validate a deployed MCP endpoint end to end, before wiring it into Claude.

    python scripts/check_remote.py https://rag.example.edu/mcp --token <MCP_AUTH_TOKEN>
    python scripts/check_remote.py https://<tunnel>.trycloudflare.com/mcp --token ...

Checks, in order: the endpoint rejects an unauthenticated request, accepts the
token, advertises the three tools, and returns real passages for a live query.
Exits non-zero on the first failure so it can gate a deploy script.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

TOOLS = {"search_textbooks", "read_context", "list_books"}
# Cloudflare's Browser Integrity Check 403s the default urllib agent (error 1010)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


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

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST",
    )
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


INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "check-remote", "version": "1.0"},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="full endpoint, including /mcp")
    ap.add_argument("--token", required=True)
    ap.add_argument("--query", default="what causes far-end crosstalk?")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ok = True

    # pasting a URL that already carries its scheme after typing one is easy to do
    # and surfaces as an opaque DNS failure, so repair it here
    url = args.url.strip()
    while "://" in url[8:]:
        url = url[url.index("://", 5) + 3:]
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
    if url != args.url.strip():
        print(f"note: cleaned up the URL -> {url}")
    args.url = url

    if not url.rstrip("/").endswith("/mcp"):
        print("note: the endpoint path usually ends in /mcp")

    print(f"checking {args.url}\n")

    print("0. endpoint is reachable")
    status, body, _ = rpc(args.url, args.token, INIT, timeout=30)
    if status is None:
        print(f"   FAIL {body}")
        print("        Could not connect at all. Check that the URL is the real one")
        print("        (not a placeholder), that the server is running on the port the")
        print("        tunnel forwards to, and that the path ends in /mcp.\n")
        return 1
    if status in (502, 503, 504):
        print(f"   FAIL HTTP {status} from the tunnel/proxy")
        print("        The tunnel is up but nothing is answering behind it. The usual")
        print("        cause is the server having started in stdio mode: MCP_TRANSPORT")
        print("        was not set in that window, so it waits on stdin and never binds")
        print("        a port. It should print a '[mcp] streamable-http listening' line.\n")
        return 1
    print(f"   OK  responded with HTTP {status}\n")

    print("1. unauthenticated request is rejected")
    status, _, _ = rpc(args.url, None, INIT)
    if status == 401:
        print("   OK  401\n")
    elif status is None:
        print("   FAIL connection dropped on the unauthenticated probe\n")
        ok = False
    else:
        print(f"   FAIL got {status}, expected 401 -- the endpoint is OPEN\n")
        ok = False

    print("2. token is accepted")
    status, msg, session = rpc(args.url, args.token, INIT)
    if status == 200 and isinstance(msg, dict) and "result" in msg:
        info = msg["result"].get("serverInfo", {})
        print(f"   OK  {info.get('name')} v{info.get('version')}\n")
    else:
        print(f"   FAIL status={status} body={msg}\n")
        return 1

    # the server expects the initialized notification before it will serve calls
    rpc(args.url, args.token,
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, session=session)

    print("3. tools are advertised")
    status, msg, _ = rpc(args.url, args.token,
                         {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                         session=session)
    names = set()
    if isinstance(msg, dict) and "result" in msg:
        names = {t["name"] for t in msg["result"].get("tools", [])}
    if TOOLS <= names:
        print(f"   OK  {sorted(names)}\n")
    else:
        print(f"   FAIL got {sorted(names)}, expected {sorted(TOOLS)}\n")
        ok = False

    print("4. a real search returns passages (first call loads models, be patient)")
    status, msg, _ = rpc(args.url, args.token, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "search_textbooks",
                   "arguments": {"query": args.query, "k": 2}},
    }, timeout=600, session=session)
    text = ""
    if isinstance(msg, dict) and "result" in msg:
        content = msg["result"].get("content", [])
        text = content[0].get("text", "") if content else ""
        if msg["result"].get("isError"):
            text = ""
    if text and "|" in text:
        print(f"   OK  {len(text)} chars")
        print(f"       {text.splitlines()[0][:100]}\n")
    else:
        print(f"   FAIL no passages: {str(msg)[:300]}\n")
        ok = False

    print("READY -- safe to add as a connector" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
