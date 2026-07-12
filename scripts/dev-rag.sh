#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/services/rag-api"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.dev.yml"
PYTHON="$SERVICE_DIR/.venv/bin/python"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

REBUILD_MILVUS="${RAG_REBUILD_MILVUS:-0}"
SKIP_FRONTEND_INSTALL="${RAG_SKIP_FRONTEND_INSTALL:-0}"
RUN_MODE="cli"
CLI_USER_ID="${RAG_CLI_USER_ID:-cli-user}"
CLI_SESSION_ID="${RAG_CLI_SESSION_ID:-cli-session}"
CLI_ID_OVERRIDDEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild-milvus|--recreate-milvus)
      REBUILD_MILVUS=1
      shift
      ;;
    --skip-frontend-install)
      SKIP_FRONTEND_INSTALL=1
      shift
      ;;
    --cli)
      RUN_MODE="cli"
      shift
      ;;
    --web)
      RUN_MODE="web"
      shift
      ;;
    --user-id)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "--user-id requires a non-empty value." >&2
        exit 2
      fi
      CLI_USER_ID="$2"
      CLI_ID_OVERRIDDEN=1
      shift 2
      ;;
    --session-id)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "--session-id requires a non-empty value." >&2
        exit 2
      fi
      CLI_SESSION_ID="$2"
      CLI_ID_OVERRIDDEN=1
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/dev-rag.sh [options]

Options:
  --rebuild-milvus          Recreate and reimport the MiniLM Milvus collection.
  --skip-frontend-install  Do not run pnpm install when frontend dependencies are missing.
  --cli                     Start the interactive RAG CLI after bootstrap (default).
  --web                     Start the FastAPI backend and Next.js frontend instead.
  --user-id ID              Development CLI identity used for memory isolation tests.
  --session-id ID           Development CLI session identity.

Environment:
  RAG_REBUILD_MILVUS=1       Same as --rebuild-milvus.
  RAG_MEMORY_PROFILE=sqlite  Use bounded SQLite memory instead of local PostgreSQL.
  RAG_MEMORY_DATABASE_URL=   Override the PostgreSQL memory database URL.
  RAG_MEMORY_WORKER_ENABLED=0  Do not start the long-memory embedding worker.
  RAG_STOP_INFRA_ON_EXIT=1   Stop Docker dependencies when this script exits.
  PNPM_REGISTRY=...          Override the npm registry used for first-time install.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_MODE" == "web" && "$CLI_ID_OVERRIDDEN" == 1 ]]; then
  echo "--user-id and --session-id are development CLI options and cannot be used with --web." >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "RAG virtual environment not found: $PYTHON" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to start Milvus." >&2
  exit 1
fi

if [[ "$RUN_MODE" == "web" ]]; then
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "pnpm is required to start the RAG frontend." >&2
    exit 1
  fi

  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof is required to detect existing RAG processes." >&2
    exit 1
  fi
fi

RAG_API_HOST="${RAG_API_HOST:-127.0.0.1}"
RAG_API_PORT="${RAG_API_PORT:-8003}"
RAG_FRONTEND_PORT="${RAG_FRONTEND_PORT:-3003}"
RAG_API_PUBLIC_HOST="${RAG_API_PUBLIC_HOST:-localhost}"
RAG_API_CORS_ORIGINS="${RAG_API_CORS_ORIGINS:-http://localhost:${RAG_FRONTEND_PORT},http://localhost:3000}"
RETRIEVAL_PROVIDER="${RETRIEVAL_PROVIDER:-milvus-hybrid}"
GROWTH_RECORDS_PATH="${GROWTH_RECORDS_PATH:-$ROOT_DIR/rawData.jsonl}"
INGEST_INPUT="${RAG_INGEST_INPUT:-$ROOT_DIR/data/processed/growth_records.text_only.jsonl}"
NEXT_PUBLIC_RAG_API_BASE_URL="${NEXT_PUBLIC_RAG_API_BASE_URL:-http://${RAG_API_PUBLIC_HOST}:${RAG_API_PORT}}"
RAG_MEMORY_PROFILE="${RAG_MEMORY_PROFILE:-postgres}"
RAG_MEMORY_WORKER_ENABLED="${RAG_MEMORY_WORKER_ENABLED:-1}"

# Pin the low-resource embedding contract so stale BGE-M3 shell variables cannot
# make a 1024-dimensional collection look compatible with MiniLM vectors.
EMBEDDING_PROVIDER="local"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BACKEND="onnx"
EMBEDDING_ONNX_MODEL_PATH="$ROOT_DIR/models/all-MiniLM-L6-v2-int8/onnx/model_quint8_avx2.onnx"
EMBEDDING_TOKENIZER_PATH="$ROOT_DIR/models/all-MiniLM-L6-v2-int8"
EMBEDDING_DIM="384"
EMBEDDING_BATCH_SIZE="4"
EMBEDDING_MAX_LENGTH="256"
EMBEDDING_DEVICE="cpu"

case "$RAG_MEMORY_PROFILE" in
  postgres)
    # The development workflow runs PostgreSQL + pgvector locally so the RAG process uses
    # the same long- and short-term-memory path as the ThinkPad deployment profile.
    MEMORY_DATABASE_URL="${RAG_MEMORY_DATABASE_URL:-postgresql://agentweb:agentweb@127.0.0.1:5432/agentweb}"
    MEMORY_CHECKPOINT_BACKEND="postgres"
    MEMORY_SEMANTIC_SEARCH_ENABLED="true"
    ;;
  sqlite)
    # Keep an explicit no-database fallback for constrained or offline development.
    MEMORY_DATABASE_URL="${RAG_MEMORY_DATABASE_URL:-${MEMORY_DATABASE_URL:-sqlite:///$ROOT_DIR/data/runtime/rag-memory.sqlite3}}"
    MEMORY_CHECKPOINT_BACKEND="${MEMORY_CHECKPOINT_BACKEND:-auto}"
    MEMORY_SEMANTIC_SEARCH_ENABLED="${MEMORY_SEMANTIC_SEARCH_ENABLED:-auto}"
    RAG_MEMORY_WORKER_ENABLED="0"
    ;;
  *)
    echo "RAG_MEMORY_PROFILE must be postgres or sqlite; got: ${RAG_MEMORY_PROFILE}" >&2
    exit 2
    ;;
esac

if [[ "$GROWTH_RECORDS_PATH" != /* ]]; then
  GROWTH_RECORDS_PATH="$ROOT_DIR/$GROWTH_RECORDS_PATH"
fi
if [[ "$INGEST_INPUT" != /* ]]; then
  INGEST_INPUT="$ROOT_DIR/$INGEST_INPUT"
fi

export RAG_API_HOST RAG_API_PORT RAG_API_CORS_ORIGINS RETRIEVAL_PROVIDER
export GROWTH_RECORDS_PATH NEXT_PUBLIC_RAG_API_BASE_URL
export EMBEDDING_PROVIDER EMBEDDING_MODEL EMBEDDING_BACKEND EMBEDDING_ONNX_MODEL_PATH
export EMBEDDING_TOKENIZER_PATH EMBEDDING_DIM EMBEDDING_BATCH_SIZE EMBEDDING_MAX_LENGTH EMBEDDING_DEVICE
export MEMORY_DATABASE_URL MEMORY_CHECKPOINT_BACKEND MEMORY_SEMANTIC_SEARCH_ENABLED

COMPOSE_SERVICES=(milvus-etcd milvus-minio milvus-standalone)
if [[ "$RAG_MEMORY_PROFILE" == "postgres" ]]; then
  COMPOSE_SERVICES=(postgres "${COMPOSE_SERVICES[@]}")
fi

echo "[rag] starting RAG dependencies"
docker compose -f "$COMPOSE_FILE" up -d "${COMPOSE_SERVICES[@]}"

if [[ "$RAG_MEMORY_PROFILE" == "postgres" ]]; then
  echo "[rag] initializing PostgreSQL long-term memory and LangGraph short-term memory"
  MEMORY_READY=0
  for attempt in $(seq 1 30); do
    if (cd "$SERVICE_DIR" && "$PYTHON" -m src.cli.rag_memory_init >/dev/null 2>&1); then
      MEMORY_READY=1
      break
    fi
    echo "[rag] waiting for PostgreSQL memory (${attempt}/30)"
    sleep 1
  done

  if [[ "$MEMORY_READY" != 1 ]]; then
    echo "PostgreSQL memory did not become ready at ${MEMORY_DATABASE_URL}." >&2
    exit 1
  fi
  (cd "$SERVICE_DIR" && "$PYTHON" -m src.cli.rag_memory_init)
fi

MILVUS_READY=0
for attempt in $(seq 1 60); do
  if (cd "$SERVICE_DIR" && "$PYTHON" -m src.cli.rag_config --check-milvus >/dev/null 2>&1); then
    MILVUS_READY=1
    break
  fi
  echo "[rag] waiting for Milvus (${attempt}/60)"
  sleep 2
done

if [[ "$MILVUS_READY" != 1 ]]; then
  echo "Milvus did not become ready at ${MILVUS_URI:-http://localhost:19530}." >&2
  echo "Check Docker Desktop and: docker compose -f infra/compose/docker-compose.dev.yml ps" >&2
  exit 1
fi

if [[ ! -f "$INGEST_INPUT" ]]; then
  echo "[rag] preparing normalized retrieval input"
  (cd "$SERVICE_DIR" && "$PYTHON" -m src.cli.rag_normalize --input "$GROWTH_RECORDS_PATH" --output-format text-only)
fi

BOOTSTRAP_ARGS=(--input "$INGEST_INPUT" --auto-recreate)
if [[ "$REBUILD_MILVUS" == 1 ]]; then
  BOOTSTRAP_ARGS+=(--recreate)
fi

echo "[rag] checking Milvus collection and data"
echo "[rag] embedding=all-MiniLM-L6-v2 backend=onnx dim=${EMBEDDING_DIM} batch=${EMBEDDING_BATCH_SIZE}"
(cd "$SERVICE_DIR" && "$PYTHON" -m src.cli.rag_bootstrap "${BOOTSTRAP_ARGS[@]}")

if [[ "$RUN_MODE" == "web" && "$SKIP_FRONTEND_INSTALL" != 1 && ! -d "$ROOT_DIR/apps/rag-platform/node_modules/next" ]]; then
  echo "[rag] installing frontend dependencies"
  (cd "$ROOT_DIR" && pnpm install --frozen-lockfile --registry="${PNPM_REGISTRY:-https://registry.npmmirror.com}")
fi

listener_pid() {
  lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

API_PID=""
WEB_PID=""
MEMORY_WORKER_PID=""
API_REUSED=0
WEB_REUSED=0

if [[ "$RUN_MODE" == "web" ]]; then
  EXISTING_API_PID="$(listener_pid "$RAG_API_PORT")"
  if [[ -n "$EXISTING_API_PID" ]]; then
    if EXISTING_API_HEALTH="$(curl --max-time 3 -fsS "http://127.0.0.1:${RAG_API_PORT}/api/rag/health")"; then
      if [[ "$RAG_MEMORY_PROFILE" == "postgres" ]] && \
        [[ "$EXISTING_API_HEALTH" != *'"memory_database":"postgres"'* || \
           "$EXISTING_API_HEALTH" != *'"short_term_backend":"postgres-checkpointer"'* ]]; then
        echo "RAG API on port ${RAG_API_PORT} is using a non-PostgreSQL memory backend." >&2
        echo "Stop pid ${EXISTING_API_PID} and rerun this script so it starts with PostgreSQL memory." >&2
        exit 1
      fi
      API_REUSED=1
      echo "[rag] reusing RAG API already listening on port ${RAG_API_PORT} (pid ${EXISTING_API_PID})"
    else
      echo "Port ${RAG_API_PORT} is occupied by pid ${EXISTING_API_PID}, but it is not a healthy RAG API." >&2
      exit 1
    fi
  fi

  EXISTING_WEB_PID="$(listener_pid "$RAG_FRONTEND_PORT")"
  if [[ -n "$EXISTING_WEB_PID" ]]; then
    if curl --max-time 3 -fsS "http://127.0.0.1:${RAG_FRONTEND_PORT}/chat" >/dev/null 2>&1; then
      WEB_REUSED=1
      echo "[rag] reusing frontend already listening on port ${RAG_FRONTEND_PORT} (pid ${EXISTING_WEB_PID})"
    else
      echo "Port ${RAG_FRONTEND_PORT} is occupied by pid ${EXISTING_WEB_PID}, but it is not a healthy RAG frontend." >&2
      exit 1
    fi
  fi
fi

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$MEMORY_WORKER_PID" ]]; then
    kill "$MEMORY_WORKER_PID" 2>/dev/null || true
    wait "$MEMORY_WORKER_PID" 2>/dev/null || true
  fi
  if [[ -n "$WEB_PID" ]]; then
    kill "$WEB_PID" 2>/dev/null || true
    wait "$WEB_PID" 2>/dev/null || true
  fi
  if [[ -n "$API_PID" ]]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  if [[ "${RAG_STOP_INFRA_ON_EXIT:-0}" == 1 ]]; then
    docker compose -f "$COMPOSE_FILE" down
  fi
}

trap cleanup EXIT INT TERM

if [[ "$RAG_MEMORY_PROFILE" == "postgres" && "$RAG_MEMORY_WORKER_ENABLED" == "1" ]]; then
  echo "[rag] starting long-term memory worker"
  (
    cd "$SERVICE_DIR"
    exec "$PYTHON" -m src.cli.rag_memory_worker
  ) &
  MEMORY_WORKER_PID=$!
fi

if [[ "$RUN_MODE" == "cli" ]]; then
  echo "[rag] starting interactive CLI"
  echo "[rag] cli user_id=${CLI_USER_ID} session_id=${CLI_SESSION_ID}"
  (
    cd "$SERVICE_DIR"
    "$PYTHON" -m src.cli.rag_chat \
      --user-id "$CLI_USER_ID" \
      --session-id "$CLI_SESSION_ID" \
      --trace \
      --top-k "${RAG_CHAT_TOP_K:-3}"
  )
  exit $?
fi

if [[ "$API_REUSED" != 1 ]]; then
  echo "[rag] starting API at http://${RAG_API_HOST}:${RAG_API_PORT}"
  (
    cd "$SERVICE_DIR"
    exec "$PYTHON" -m uvicorn src.main:app --host "$RAG_API_HOST" --port "$RAG_API_PORT"
  ) &
  API_PID=$!

  API_READY=0
  for attempt in $(seq 1 30); do
    if curl --max-time 3 -fsS "http://127.0.0.1:${RAG_API_PORT}/api/rag/health" >/dev/null 2>&1; then
      API_READY=1
      break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
      wait "$API_PID" || true
      echo "RAG API exited before becoming ready." >&2
      exit 1
    fi
    sleep 1
  done

  if [[ "$API_READY" != 1 ]]; then
    echo "RAG API did not become ready on port ${RAG_API_PORT}." >&2
    exit 1
  fi
fi

if [[ "$WEB_REUSED" != 1 ]]; then
  echo "[rag] starting frontend at http://localhost:${RAG_FRONTEND_PORT}/chat"
  (
    cd "$ROOT_DIR/apps/rag-platform"
    exec pnpm exec next dev -p "$RAG_FRONTEND_PORT"
  ) &
  WEB_PID=$!

  WEB_READY=0
  for attempt in $(seq 1 30); do
    if curl --max-time 3 -fsS "http://127.0.0.1:${RAG_FRONTEND_PORT}/chat" >/dev/null 2>&1; then
      WEB_READY=1
      break
    fi
    if ! kill -0 "$WEB_PID" 2>/dev/null; then
      wait "$WEB_PID" || true
      echo "RAG frontend exited before becoming ready." >&2
      exit 1
    fi
    sleep 1
  done

  if [[ "$WEB_READY" != 1 ]]; then
    echo "RAG frontend did not become ready on port ${RAG_FRONTEND_PORT}." >&2
    exit 1
  fi
fi

echo "[rag] ready: http://localhost:${RAG_FRONTEND_PORT}/chat"
echo "[rag] press Ctrl-C to stop API and frontend"

if [[ -z "$API_PID" && -z "$WEB_PID" ]]; then
  echo "[rag] both services were already running"
  exit 0
fi

while true; do
  if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" || true
    echo "RAG API stopped." >&2
    exit 1
  fi
  if [[ -n "$WEB_PID" ]] && ! kill -0 "$WEB_PID" 2>/dev/null; then
    wait "$WEB_PID" || true
    echo "RAG frontend stopped." >&2
    exit 1
  fi
  if [[ -n "$MEMORY_WORKER_PID" ]] && ! kill -0 "$MEMORY_WORKER_PID" 2>/dev/null; then
    wait "$MEMORY_WORKER_PID" || true
    echo "RAG long-term memory worker stopped." >&2
    exit 1
  fi
  sleep 1
done
