#!/usr/bin/env bash
# One-command deploy: generates the secrets on first run, builds the index from
# whatever is in original/, and serves it. Safe to re-run -- it is how you pick up
# a newly added PDF, and it never regenerates secrets that already exist.
#
#   ./deploy.sh                 local / behind a tunnel
#   ./deploy.sh --tls           also run Caddy for a public HTTPS domain
set -euo pipefail
cd "$(dirname "$0")"

command -v docker >/dev/null || { echo "docker is not installed: https://docs.docker.com/get-docker/" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose v2 is required" >&2; exit 1; }

secret() { python3 -c 'import secrets;print(secrets.token_urlsafe(32))' 2>/dev/null || openssl rand -base64 32 | tr -d '/+=' ; }

if [ ! -f .env ]; then
  cp .env.example .env
  # A passphrase people have to type gets words, not base64. Everything machines
  # read gets full entropy.
  words=$(LC_ALL=C tr -dc 'a-z' </dev/urandom | head -c 12)
  sed -i.bak "s|^MCP_TEAM_PASSWORD=.*|MCP_TEAM_PASSWORD=${words:0:4}-${words:4:4}-${words:8:4}|" .env
  sed -i.bak "s|^QDRANT_API_KEY=.*|QDRANT_API_KEY=$(secret)|" .env
  sed -i.bak "s|^MCP_AUTH_TOKEN=.*|MCP_AUTH_TOKEN=$(secret)|" .env
  rm -f .env.bak
  echo "wrote .env with fresh secrets"
fi

profile=()
[ "${1:-}" = "--tls" ] && profile=(--profile tls)

echo "building the index and starting the stack (first run downloads ~2.8 GB of models)"
docker compose "${profile[@]}" up -d --build

echo
echo "--- connect Claude to it ----------------------------------------------"
grep -E '^(MCP_PUBLIC_URL|MCP_TEAM_PASSWORD)=' .env
echo "Add <MCP_PUBLIC_URL>/mcp as a custom connector; the passphrase is the login."
echo
echo "logs:    docker compose logs -f mcp builder"
echo "rebuild: drop a PDF in original/ and run ./deploy.sh again"
