"""
End-to-end check that the MCP server speaks clean JSON-RPC over stdio.

This is the failure mode that matters for Claude Desktop: model loaders and HTTP
clients print to stdout, which corrupts the protocol stream. Running the server
as a real subprocess is the only way to catch it.

stdin is held open for the whole session -- the server cancels in-flight tool
calls when stdin hits EOF, exactly as a real client would avoid.

    python scripts/test_mcp_stdio.py
"""
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TIMEOUT = 600  # first tool call pays for model load


def reader(pipe, q):
    for line in pipe:
        q.put(line)
    q.put(None)


def main():
    proc = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / "mcp_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    q = queue.Queue()
    threading.Thread(target=reader, args=(proc.stdout, q), daemon=True).start()
    errs = []
    threading.Thread(
        target=lambda: errs.extend(proc.stderr), daemon=True
    ).start()

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

    send({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "stdio-test", "version": "1.0"},
        },
    })
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
        tools = sorted(t["name"] for t in tl.get("result", {}).get("tools", []))
        print(f"  tools/list  -> {tools}")
        if tools != ["list_books", "read_context", "search_textbooks"]:
            ok = False
    else:
        print("  tools/list  -> MISSING")
        ok = False

    def call(cid, name, args):
        nonlocal ok
        send({"jsonrpc": "2.0", "id": cid, "method": "tools/call",
              "params": {"name": name, "arguments": args}})
        msg = await_id(cid)
        if not msg:
            print(f"  {name:<16} -> MISSING")
            ok = False
            return None
        result = msg.get("result", {})
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        flag = result.get("isError")
        print(f"  {name:<16} -> {len(text)} chars, isError={flag}")
        if text:
            print(f"      {text.splitlines()[0][:96]}")
        if not text or flag:
            ok = False
        return text

    call(3, "list_books", {})
    hit = call(4, "search_textbooks",
               {"query": "what is the loss tangent of a dielectric?", "k": 2})
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

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
