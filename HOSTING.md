# Hosting the MCP server

How to get the corpus in front of Claude, end to end. `README.md` covers the
retrieval pipeline and adding documents; this file is only about serving it.

## 0. Do you need to host anything?

If someone else is already running it, **you install nothing**. Open Claude, add a
custom connector pointing at their URL (e.g. `https://rag.sipi-lab.org/mcp`), enter
the team passphrase. No clone, no Python, no model downloads. Query embedding and
reranking happen on the server.

Host something only if you are the one serving, or you want a private local copy.

## 1. Pick a mode

| | Who reaches it | What runs | Survives reboot |
|---|---|---|---|
| **A. Claude Desktop (stdio)** | you, on this machine | nothing persistent | n/a |
| **B. Native + tunnel** | anyone with the URL | `rag.py serve` + tunnel | with `autostart` |
| **C. Docker** | anyone with the URL | compose stack | yes, supervised |

**A** is the least moving parts. **B** is a laptop or workstation you already use —
it is what `rag.sipi-lab.org` runs on today. **C** is for a machine that is meant to
stay up; only it restarts the server automatically after a crash.

Whatever you pick, the index must be built first (step 2).

---

## 2. Build the index (all modes)

```bash
git clone https://github.com/thomasjtHe/rag_project.git
cd rag_project
python rag.py setup      # .venv, torch matched to your GPU, dependencies
python rag.py build      # original/*.pdf -> vectors
```

The PDFs and their extracted text are committed, so a fresh clone can build
without hunting for sources. Expect the first run to download ~2.8 GB of model
weights, and the embedding pass to take a while on CPU (minutes on a GPU).

Check it landed:

```bash
python rag.py status
python rag.py search "what causes far-end crosstalk?"
```

Docker (mode C) does this for you — skip to §5.

---

## 3. Mode A — Claude Desktop, local, no network

Merge `claude_desktop_config.example.json` into your Claude Desktop config, with
both paths made absolute:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

Point `command` at the venv's interpreter (`.venv\Scripts\python.exe`, or
`.venv/bin/python`), **not** a bare `python` — that one has none of the
dependencies. Restart Claude Desktop.

Verify independently of the client:

```bash
python rag.py doctor
```

This runs the server as a real subprocess over stdio and fails if anything
pollutes stdout, which is the usual reason a server appears in Claude with no
tools. The first tool call loads the models and takes about a minute.

---

## 4. Mode B — native server behind a tunnel

### 4.1 Set the secrets

`python rag.py setup` created `.env` from `.env.example`. Set at least:

```ini
MCP_TEAM_PASSWORD=four-word-phrase     # what teammates type to sign in
MCP_PUBLIC_URL=https://rag.example.org # the URL, no /mcp suffix
```

`MCP_TEAM_PASSWORD` is what makes the server speak OAuth, which is the only mode
Claude's connector UI accepts. Without it the server refuses to expose itself.

### 4.2 Get a public hostname

A **named Cloudflare tunnel** gives a permanent URL, so the connector is added
once and never touched again:

```bash
cloudflared tunnel login
cloudflared tunnel create si-rag
cloudflared tunnel route dns si-rag rag.example.org
```

The tunnel's *name* (`si-rag`) and its *hostname* (`rag.example.org`) are
different things — `rag.py serve` takes both, and needs both.

For a throwaway URL instead, omit `--name`/`--domain` below; the URL changes on
every restart, so the connector must be re-added each time.

### 4.3 Serve

```bash
python rag.py serve --tunnel cloudflare --name si-rag --domain rag.example.org
```

It starts the tunnel first (the server needs its public hostname before it binds,
to allow that `Host` header through the SDK's DNS-rebinding protection), then the
server. Ctrl+C stops both.

### 4.4 Keep it up without a terminal

```bash
python rag.py autostart --tunnel cloudflare --name si-rag --domain rag.example.org
```

Windows: writes a Startup-folder script that launches windowless at logon —
`--status` shows it, `--remove` uninstalls it. Logs go to `data/serve.log`.
Linux/macOS: prints a systemd user unit to install.

Caveat: this starts at **logon**, not boot, and does not restart the server if it
crashes. For real supervision use mode C.

---

## 5. Mode C — Docker

Needs Docker Engine + Compose v2. One command:

```bash
./deploy.sh          # Linux/macOS
.\deploy.ps1         # Windows
```

It generates `.env` with fresh secrets on first run, starts Qdrant, builds the
index from `original/`, and serves. Re-run it to pick up a new PDF; it re-embeds
only what changed and never regenerates existing secrets.

The stack is four services: `qdrant`, `builder` (runs the pipeline and exits),
`mcp`, and `caddy` (only under the `tls` profile). `mcp` waits for `builder` to
finish, so the endpoint is never live against a half-built index.

`mcp` publishes to `127.0.0.1:8000` only. Expose it one of three ways:

- **Tunnel** — run `cloudflared` on the host against `http://localhost:8000`, and
  set `MCP_PUBLIC_URL` to the tunnel hostname. Leave `DOMAIN` empty.
- **Caddy** — only if the host has its own public DNS record and ports 80+443 open.
  Set `DOMAIN`, then `./deploy.sh --tls`. Caddy obtains the certificate.
- **Reverse proxy you already run** — change the port mapping to `8000:8000` and
  set `MCP_PUBLIC_URL` to whatever you terminate.

After editing `.env`, re-run `./deploy.sh` to apply it.

---

## 6. Verify before telling anyone

```bash
python rag.py doctor --url https://rag.example.org/mcp
```

With no `--token` this checks the OAuth path: that the endpoint answers, that
`/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`
resolve **at the public hostname**, that the metadata carries a registration
endpoint, and that the login page loads. Those are the exact documents Claude's
connector UI reads. Wait for `PASS`.

Then in Claude: add a custom connector with `https://rag.example.org/mcp`, and sign
in with `MCP_TEAM_PASSWORD`. Do not use "Continue anyway" or enter a client ID —
the server does dynamic client registration itself.

## 7. Environment reference

| variable | who reads it | notes |
|---|---|---|
| `MCP_TEAM_PASSWORD` | server | Enables OAuth. Required for Claude's connector UI. |
| `MCP_PUBLIC_URL` | server | The public URL, no `/mcp`. OAuth metadata is built from it — wrong value breaks sign-in. |
| `MCP_AUTH_TOKEN` | server | Static bearer token for scripted clients. Claude's UI will not accept it. |
| `MCP_ALLOWED_HOSTS` | server | Extra `Host` headers to accept. Derived from `MCP_PUBLIC_URL` by default. |
| `DOMAIN` | **Caddy only** | Only used by the `tls` compose profile. Leave empty otherwise. |
| `QDRANT_URL` | scripts | Shared Qdrant. Unset = local `data/qdrant` file. |
| `QDRANT_API_KEY` | Qdrant + scripts | Qdrant's only access control; it has no user accounts. |
| `RAG_RERANK` | server | `0` drops the cross-encoder: 3–5x faster on CPU, worse results. |
| `ANTHROPIC_API_KEY` | `rag.py chat` | Not needed to serve. |

## 8. Operations

- **Add a document** — drop the PDF in `original/`, then `python rag.py build` (or
  `./deploy.sh`). Only the new book is embedded. Check `corpus.json` afterwards for
  the auto-detected contents/index pages.
- **Rotate the passphrase** — change `MCP_TEAM_PASSWORD`, restart. Everyone
  reconnects once; there is no grace period with a single shared secret.
- **Logs** — `data/serve.log` (modes B), `docker compose logs -f mcp builder` (C).
- **Move to another host** — stop the tunnel on the old host first. Two connectors
  on one named tunnel make Cloudflare balance between them, and the one without a
  working origin fails about half of all requests.

## 9. Troubleshooting

Every problem seen so far in practice has been transport, not auth. Work down the
chain: is the server up, is the tunnel carrying traffic, is the URL right.

| Symptom | Cause | Fix |
|---|---|---|
| Claude: *"couldn't determine how this server signs in"* | Discovery unreachable — usually the tunnel has no origin | `rag.py doctor --url ...`; if step 0 times out the tunnel is the problem |
| Claude: *"couldn't register with the sign-in service"* | Endpoint reachable but server behind it is gone | Check the server is running; `doctor` should reach step 3 |
| Cloudflare **error 1033** | Tunnel up, nothing behind it | Server not running, or it lost the port bind |
| Works, then stops after a while | An unread tunnel log pipe filling and blocking | Fixed in `rag.py`; make sure you are not running `cloudflared` by hand without draining it |
| Intermittent failures, ~50% | Two connectors registered for one named tunnel | Kill the stale `cloudflared` |
| `port 8000 is already in use` | A previous server is still running | Stop it (the message names the PID) or `--port` |
| Server logs `Errno 10048` | Same, from an older build that did not preflight | Update; `serve` now refuses to start |
| `dependency failed to start: ... is unhealthy` | Old Qdrant healthcheck using bash-only `/dev/tcp` under dash | Fixed; `git pull`, `docker compose down`, redeploy |
| `set QDRANT_API_KEY in .env` | Ran bare `docker compose up` with no `.env` | Use `./deploy.sh` |
| Tools missing in Claude Desktop | Something printed to stdout, corrupting JSON-RPC | `python rag.py doctor` reports the offending lines |

## 10. Before opening it up

The index holds substantial verbatim text from copyrighted textbooks, including
one retrieved through a university subscription. A personal local index is one
thing; an org-wide connector puts that text in front of everyone in the
organization, which is a different act. Worth settling with your library's
licensing contact first.

`MCP_TEAM_PASSWORD` is a single shared secret with no per-user identity and no way
to revoke one person without rotating for everyone. Qdrant has no accounts at all.
Keep both off the public internet — VPN, campus network, or a tunnel with access
controls — and put a real IdP in front if you need per-user audit.
