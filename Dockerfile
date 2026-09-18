# CPU-only image: cloud VMs rarely have a GPU, and query-time embedding is one
# short sequence -- it does not need one. The ~2.8 GB of model weights are NOT
# baked in; they download on first run into a mounted volume (see compose), so
# the image stays small and the weights survive a rebuild.
#
# One image serves both roles in the stack: the `builder` service runs the
# ingest pipeline and exits, the `mcp` service serves. Same code, same versions,
# so an index can never be built by a different release than the one querying it.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    RAG_NO_REEXEC=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# torch first, from the CPU index -- otherwise pip pulls the CUDA build (~2.5 GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY scripts/ /app/scripts/
COPY rag.py corpus.json /app/

# fail fast if a network deployment forgot its shared secret
RUN printf '%s\n' \
  '#!/bin/sh' \
  'set -e' \
  'if [ -z "$MCP_AUTH_TOKEN" ] && [ -z "$MCP_TEAM_PASSWORD" ]; then' \
  '  echo "refusing to start: set MCP_TEAM_PASSWORD or MCP_AUTH_TOKEN (the endpoint would be open)" >&2' \
  '  exit 1' \
  'fi' \
  'exec python /app/scripts/mcp_server.py' \
  > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp

EXPOSE 8000
CMD ["/usr/local/bin/start.sh"]
