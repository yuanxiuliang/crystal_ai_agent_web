#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.x1c.yml"
ENV_FILE="$ROOT_DIR/infra/compose/.env.x1c"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing deployment environment file: $ENV_FILE" >&2
  echo "Create it from infra/compose/.env.x1c.example and replace all CHANGE_ME values." >&2
  exit 1
fi

if grep -q '^.*=CHANGE_ME' "$ENV_FILE"; then
  echo "Deployment environment still contains CHANGE_ME placeholders: $ENV_FILE" >&2
  exit 1
fi

compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

build_args=(build)
if [[ "${RAG_DEPLOY_PULL_BASE_IMAGES:-1}" == "1" ]]; then
  build_args+=(--pull)
fi

echo "[deploy] validating production Compose configuration"
"${compose[@]}" config -q

echo "[deploy] building CPU-only API and Web images through configured mirrors"
"${compose[@]}" "${build_args[@]}"

echo "[deploy] starting private RAG services"
"${compose[@]}" up -d --remove-orphans

echo "[deploy] waiting for bootstrap, API, and Web health checks"
for attempt in $(seq 1 120); do
  if "$ROOT_DIR/scripts/check-x1c-rag.sh" >/dev/null 2>&1; then
    "$ROOT_DIR/scripts/check-x1c-rag.sh"
    exit 0
  fi
  sleep 2
done

echo "[deploy] deployment did not become healthy in four minutes." >&2
"${compose[@]}" ps >&2
"${compose[@]}" logs --tail=100 rag-bootstrap rag-api rag-web >&2
exit 1
