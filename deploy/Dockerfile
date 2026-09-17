# CPU-only image: cloud VMs rarely have a GPU, and query-time embedding is one
# short sequence -- it does not need one. The ~2.8 GB of model weights are NOT
# baked in; they download on first run into a mounted volume (see compose), so
# the image stays small and the weights survive a rebuild.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# torch first, from the CPU index -- otherwise pip pulls the CUDA build (~2.5 GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

RUN pip install \
      "numpy<2" \
      FlagEmbedding \
      "qdrant-client[fastembed]" \
      "mcp>=2.0"

COPY scripts/ /app/scripts/

# fail fast if the deployment forgot the shared secret
RUN printf '%s\n' \
  '#!/bin/sh' \
  'set -e' \
  'if [ -z "$MCP_AUTH_TOKEN" ]; then' \
  '  echo "refusing to start: MCP_AUTH_TOKEN is unset (the endpoint would be open)" >&2' \
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
