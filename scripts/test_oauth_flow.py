"""
Drive the full OAuth flow against a locally running server, the way Claude does.

    # terminal 1
    MCP_TEAM_PASSWORD=letmein MCP_TRANSPORT=streamable-http MCP_PORT=8020 \
    MCP_PUBLIC_URL=http://127.0.0.1:8020 python scripts/mcp_server.py

    # terminal 2
    python scripts/test_oauth_flow.py http://127.0.0.1:8020 --password letmein

Steps: discover metadata, register a client dynamically, run the authorize
redirect, post the passphrase, exchange the code (with PKCE), then call a tool
with the resulting access token.
"""
import argparse
import base64
import hashlib
import json
import secrets
import time
import sys
import urllib.error
import urllib.parse
import urllib.request

REDIRECT = "http://localhost:9999/callback"
# Cloudflare's Browser Integrity Check 403s the default urllib agent (error 1010)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def http(url, data=None, headers=None, method=None, allow_redirect=True,
         as_json=False):
    body = None
    if isinstance(data, dict) and not as_json:
        body = urllib.parse.urlencode(data).encode()
        headers = {**(headers or {}),
                   "Content-Type": "application/x-www-form-urlencoded"}
    elif data is not None:
        body = json.dumps(data).encode()
        headers = {**(headers or {}), "Content-Type": "application/json"}

    headers = {**(headers or {}), "User-Agent": USER_AGENT}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = (urllib.request.build_opener()
              if allow_redirect else urllib.request.build_opener(NoRedirect))
    try:
        with opener.open(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. http://127.0.0.1:8020")
    ap.add_argument("--password", required=True)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    ok = True

    print("1. protected-resource metadata")
    st, body, _ = http(f"{base}/.well-known/oauth-protected-resource")
    print(f"   {st}  {body[:150] if st == 200 else body[:200]}\n")
    if st != 200:
        ok = False

    print("2. authorization-server metadata")
    st, body, _ = http(f"{base}/.well-known/oauth-authorization-server")
    meta = json.loads(body) if st == 200 else {}
    if st == 200:
        print(f"   200  registration={bool(meta.get('registration_endpoint'))} "
              f"authorize={bool(meta.get('authorization_endpoint'))} "
              f"token={bool(meta.get('token_endpoint'))}\n")
    elif st == 404:
        print("   FAIL 404 -- no authorization server on this endpoint.")
        print("        The server is running in bearer-token mode. OAuth endpoints")
        print("        only exist when MCP_TEAM_PASSWORD is set, so restart it with")
        print("        that variable; its banner should read auth=oauth, not auth=token.\n")
        return 1
    else:
        print(f"   FAIL {st} {body[:200]}\n")
        return 1

    print("3. dynamic client registration")
    st, body, _ = http(meta["registration_endpoint"], as_json=True, data={
        "client_name": "flow-test",
        "redirect_uris": [REDIRECT],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
        "scope": "read",
    })
    if st not in (200, 201):
        print(f"   FAIL {st} {body[:300]}\n")
        return 1
    client = json.loads(body)
    print(f"   {st}  client_id={client['client_id'][:24]}...\n")

    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    state = secrets.token_urlsafe(12)

    print("4. authorize -> login page")
    q = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client["client_id"],
        "redirect_uri": REDIRECT,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "read",
        "resource": base,
    })
    st, body, hdrs = http(f"{meta['authorization_endpoint']}?{q}",
                          allow_redirect=False)
    loc = hdrs.get("Location") or hdrs.get("location")
    if st in (302, 303, 307) and loc and "/login" in loc:
        print(f"   {st}  -> {loc}\n")
    else:
        print(f"   FAIL {st} loc={loc} {body[:200]}\n")
        return 1
    rk = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query).get("rk", [""])[0]

    print("5. wrong passphrase is rejected")
    st, _, _ = http(f"{base}/login",
                    data={"rk": rk, "pw": "definitely-wrong", "who": "tester"},
                    allow_redirect=False)
    if st == 401:
        print("   OK  401\n")
    else:
        print(f"   FAIL got {st}, expected 401\n")
        ok = False

    print("6. correct passphrase -> authorization code")
    st, body, hdrs = http(f"{base}/login",
                          data={"rk": rk, "pw": args.password, "who": "tester"},
                          allow_redirect=False)
    loc = hdrs.get("Location") or hdrs.get("location")
    if st not in (302, 303) or not loc:
        print(f"   FAIL {st} {body[:200]}\n")
        return 1
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
    code = qs.get("code", [""])[0]
    print(f"   {st}  code={code[:16]}...  state_ok={qs.get('state', [''])[0] == state}\n")

    print("7. exchange code for a token (PKCE)")
    st, body, _ = http(meta["token_endpoint"], data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "client_id": client["client_id"],
        "client_secret": client.get("client_secret", ""),
        "code_verifier": verifier,
        "resource": base,
    })
    if st != 200:
        print(f"   FAIL {st} {body[:300]}\n")
        return 1
    tok = json.loads(body)
    access = tok["access_token"]
    print(f"   200  access_token={access[:16]}...  refresh={bool(tok.get('refresh_token'))}\n")

    print("8. call a tool with the access token")
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18",
    }
    st, body, hdrs = http(f"{base}/mcp", as_json=True, data={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "flow-test", "version": "1"}},
    }, headers=headers)
    session = hdrs.get("mcp-session-id")
    if st != 200:
        print(f"   FAIL initialize {st} {body[:200]}\n")
        return 1
    headers["mcp-session-id"] = session
    http(f"{base}/mcp", as_json=True,
         data={"jsonrpc": "2.0", "method": "notifications/initialized"},
         headers=headers)
    st, body, _ = http(f"{base}/mcp", as_json=True, data={
        "jsonrpc": "2.0", "id": 2, "method": "tools/list",
    }, headers=headers)
    names = []
    for line in body.splitlines():
        if line.startswith("data:"):
            msg = json.loads(line[5:].strip())
            names = [t["name"] for t in msg.get("result", {}).get("tools", [])]
    if names:
        print(f"   200  tools={sorted(names)}\n")
    else:
        print(f"   FAIL {st} {body[:200]}\n")
        ok = False

    print("9. real search (cold: loads ~2.8GB of models on first call)")
    t0 = time.time()
    st, body, _ = http(f"{base}/mcp", as_json=True, data={
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "search_textbooks",
                   "arguments": {"query": "what causes far-end crosstalk?", "k": 2}},
    }, headers=headers)
    elapsed = time.time() - t0
    text = ""
    for line in (body or "").splitlines():
        if line.startswith("data:"):
            try:
                msg = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            content = msg.get("result", {}).get("content") or [{}]
            text = content[0].get("text", "")
    if text and "|" in text:
        print(f"   200  {elapsed:.1f}s  {text.splitlines()[0][:80]}")
    else:
        print(f"   FAIL {st} after {elapsed:.1f}s: {str(body)[:200]}")
        if elapsed > 95:
            print("        Over ~100s: Cloudflare's free-plan proxy timeout (error 524).")
            print("        Warm the models with one direct localhost call after starting,")
            print("        or keep the server running so later calls stay fast.")
        ok = False
    print()

    print("PASS -- Claude's OAuth flow will work" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
