# Org-wide connector deployment

Runs the retriever as a remote MCP server so everyone in the Claude organization
can use it without installing anything — no repo, no Python, no model downloads.

    internet ──► Caddy (TLS) ──► mcp (streamable-http, bearer auth) ──► qdrant
                                        └── model_cache volume (~2.8 GB)

## Before provisioning anything: check the auth handshake

The server authenticates with a **static bearer token**. Whether Claude's custom
connector UI lets you supply one — as opposed to requiring a full OAuth 2.0
authorization server — is the single thing that decides how much work this is,
and it is worth confirming in under an hour rather than after you have paid for a
VM.

Do this first, with a tunnel from your laptop:

```bash
# terminal 1 — run the server locally with auth on
export MCP_AUTH_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
export MCP_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 MCP_PORT=8000
export MCP_PUBLIC_URL=https://<whatever-the-tunnel-gives-you>
python scripts/mcp_server.py

# terminal 2 — expose it over public HTTPS
cloudflared tunnel --url http://localhost:8000
```

Take the tunnel's `https://...` URL, add `/mcp`, and try adding it in Claude as a
custom connector. If it accepts a bearer token, the rest of this file is a
straightforward deployment. If it insists on OAuth, stop and read
"If OAuth is required" below before going further.

## Deploy

Needs a host with a public DNS name pointing at it and inbound 80 + 443 — Caddy
needs both to issue a Let's Encrypt certificate.

```bash
cp deploy/.env.example deploy/.env      # set DOMAIN, MCP_AUTH_TOKEN, QDRANT_API_KEY
docker compose -f deploy/docker-compose.yml up -d --build
```

The first request downloads ~2.8 GB of model weights into the `model_cache`
volume and takes a few minutes. Later restarts reuse it.

### Load the index

Qdrant is not published to the host — it is only on the internal compose network.
Tunnel to it for the one-off ingest:

```bash
ssh -L 6333:localhost:6333 user@host          # with qdrant's port temporarily published
QDRANT_URL=http://localhost:6333 QDRANT_API_KEY=<key> \
    python scripts/ingest_qdrant.py --recreate
```

Alternatively copy `data/chunks.jsonl` to the host and run the ingest there,
inside the compose network.

### Verify

```bash
# expect 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://<domain>/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"c","version":"1"}}}'

# expect 200 and a serverInfo payload
curl -s -X POST https://<domain>/mcp -H "Authorization: Bearer $MCP_AUTH_TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"c","version":"1"}}}'
```

### Add it in Claude

As a Team/Enterprise admin, add a custom connector in the organization settings
with URL `https://<domain>/mcp`. Teammates then enable `signal-integrity-books`
and get the three tools. Exact menu labels move around between releases — look
for Connectors in organization/admin settings.

## If OAuth is required

The MCP SDK supports it: `MCPServer` takes `auth_server_provider` alongside the
`token_verifier` this deployment uses. Two realistic routes:

- Front the server with an OAuth-aware proxy (oauth2-proxy, or your university's
  SSO if it speaks OIDC). This also gets you per-user identity, which the shared
  token does not.
- Implement `OAuthAuthorizationServerProvider` in `scripts/mcp_server.py`.
  More code, no dependency on institutional SSO.

The shared-token design is deliberately the floor, not the target: one secret for
the whole team, rotated by restarting with a new value, with no per-user identity
and no way to revoke one member without rotating for everyone.

## Operations

- **Rotate the token:** change `MCP_AUTH_TOKEN` in `deploy/.env`, then
  `docker compose -f deploy/docker-compose.yml up -d mcp`. Every client must be
  updated — there is no grace period with a single shared secret.
- **Update the corpus:** re-run `chunk_corpus.py` locally, then `ingest_qdrant.py`
  against the deployed Qdrant. The MCP server needs no restart.
- **Logs:** `docker compose -f deploy/docker-compose.yml logs -f mcp caddy`
- **Performance:** CPU-only. Query embedding is ~1-3 s; the cross-encoder reranker
  adds a few seconds on 30 candidates. Set `RAG_RERANK=0` to trade result quality
  for latency, or give the VM more cores — `OMP_NUM_THREADS` in the Dockerfile is
  set to 4.

## Before you switch it on for the org

The index holds substantial verbatim text from three copyrighted textbooks,
including one retrieved through a university subscription. An org-wide connector
puts that in front of everyone in the organization, which is a different act from
personal research use. Worth clearing with your library's licensing contact
first.

## Status

Verified on the development machine: bearer auth (401 without a token, 401 with a
wrong one, 200 with the right one), the streamable-http endpoint, and the
host/port binding that makes the container reachable.

Not verified: the Docker image build and compose stack — Docker was not installed
on the machine where this was written. Expect to debug the build once, on the
host.
