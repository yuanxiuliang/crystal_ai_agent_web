#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-agentweb-rag-e2e}"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.e2e.yml"
TEST_IMAGE="${RAG_E2E_TEST_IMAGE:-agentweb-rag-api-test:e2e}"
API_IMAGE="${RAG_E2E_API_IMAGE:-agentweb-rag-api:e2e}"
WEB_PORT="${RAG_E2E_WEB_PORT:-13003}"
E2E_PLATFORM="${RAG_E2E_PLATFORM:-linux/amd64}"

# Docker Hub is the portable default. Set RAG_E2E_IMAGE_PROXY to a registry proxy such as
# docker.1ms.run in domestic networks. Every image remains individually overrideable for a
# private registry or a locally preloaded cache.
if [[ -n "${RAG_E2E_IMAGE_PROXY:-}" ]]; then
  proxy="${RAG_E2E_IMAGE_PROXY%/}"
  export RAG_E2E_API_BASE_IMAGE="${RAG_E2E_API_BASE_IMAGE:-$proxy/library/python:3.12-slim-bookworm}"
  export RAG_E2E_WEB_BASE_IMAGE="${RAG_E2E_WEB_BASE_IMAGE:-$proxy/library/node:22-bookworm-slim}"
  export RAG_E2E_POSTGRES_IMAGE="${RAG_E2E_POSTGRES_IMAGE:-$proxy/pgvector/pgvector:pg16}"
  export RAG_E2E_ETCD_IMAGE="${RAG_E2E_ETCD_IMAGE:-$proxy/quay.io/coreos/etcd:v3.5.18}"
  export RAG_E2E_MINIO_IMAGE="${RAG_E2E_MINIO_IMAGE:-$proxy/minio/minio:RELEASE.2024-12-18T13-15-44Z}"
  export RAG_E2E_MILVUS_IMAGE="${RAG_E2E_MILVUS_IMAGE:-$proxy/milvusdb/milvus:v2.6.0}"
fi

for required in RAG_E2E_LLM_BASE_URL RAG_E2E_LLM_API_KEY RAG_E2E_LLM_MODEL; do
  if [[ -z "${!required:-}" ]]; then
    echo "[e2e] missing required real LLM configuration: $required" >&2
    exit 2
  fi
done

compose=(docker compose --project-name "$COMPOSE_PROJECT_NAME" --file "$COMPOSE_FILE")

cleanup() {
  status=$?
  if [[ "$status" -ne 0 ]]; then
    "${compose[@]}" ps >&2 || true
    "${compose[@]}" logs --no-color --tail=160 >&2 || true
  fi
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

cd "$ROOT_DIR"
"${compose[@]}" config -q

echo "[e2e] building production-like API and Web images"
"${compose[@]}" build rag-api rag-web
docker build \
  --platform "$E2E_PLATFORM" \
  --build-arg "BASE_IMAGE=$API_IMAGE" \
  --tag "$TEST_IMAGE" \
  --file "$ROOT_DIR/infra/docker/rag-api-test.Dockerfile" \
  "$ROOT_DIR"

echo "[e2e] starting disposable PostgreSQL, Milvus, API, and Web services"
"${compose[@]}" up -d

for _ in $(seq 1 90); do
  if "${compose[@]}" exec -T rag-api \
    python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8003/api/rag/health', timeout=3).status == 200" \
    >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

"${compose[@]}" exec -T rag-api \
  python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8003/api/rag/health', timeout=3).status == 200"

for _ in $(seq 1 45); do
  if curl --fail --silent --show-error --max-time 3 "http://127.0.0.1:${WEB_PORT}/login" >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${WEB_PORT}/login" >/dev/null

echo "[e2e] rerunning bootstrap to verify collection idempotency"
bootstrap_output="$("${compose[@]}" run --rm --no-deps rag-bootstrap)"
printf '%s\n' "$bootstrap_output"
grep -Fq "catalog_status=ready" <<<"$bootstrap_output"
grep -Fq "collection is ready; skipping import" <<<"$bootstrap_output"

echo "[e2e] running real-LLM API contracts against the disposable stack"
docker run --rm \
  --network "${COMPOSE_PROJECT_NAME}_edge" \
  --mount "type=bind,src=$ROOT_DIR/e2e/api,dst=/opt/agentweb/e2e/api,readonly" \
  -e RAG_E2E_API_BASE_URL=http://rag-api:8003 \
  -e MEMORY_DATABASE_URL=postgresql://e2e:e2e@postgres:5432/e2e \
  -e MEMORY_CHECKPOINT_BACKEND=postgres \
  "$TEST_IMAGE" \
  pytest -q -p no:cacheprovider /opt/agentweb/e2e/api

echo "[e2e] running browser workflow checks"
E2E_BASE_URL="http://127.0.0.1:${WEB_PORT}" pnpm exec playwright test --config=e2e/playwright.config.ts

echo "[e2e] all disposable RAG end-to-end checks passed"
