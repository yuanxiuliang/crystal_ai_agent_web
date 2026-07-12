#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../services/rag-api"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "rag-api: missing .venv/bin/python; set up the RAG service environment first." >&2
  exit 1
fi

exec .venv/bin/python -m uvicorn src.main:app \
  --host "${RAG_API_HOST:-127.0.0.1}" \
  --port "${RAG_API_PORT:-8003}"
