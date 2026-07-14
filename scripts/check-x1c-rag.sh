#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.x1c.yml"
ENV_FILE="$ROOT_DIR/infra/compose/.env.x1c"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing deployment environment file: $ENV_FILE" >&2
  exit 1
fi

compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

echo "[check] validating container states"
"${compose[@]}" ps

echo "[check] verifying PostgreSQL"
"${compose[@]}" exec -T postgres pg_isready -U "$(sed -n 's/^POSTGRES_USER=//p' "$ENV_FILE" | tail -n 1)" \
  -d "$(sed -n 's/^POSTGRES_DB=//p' "$ENV_FILE" | tail -n 1)" >/dev/null

echo "[check] verifying RAG API memory profile"
health="$(curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8003/api/rag/health)"
printf '%s\n' "$health"
[[ "$health" == *'"status":"ok"'* ]]
[[ "$health" == *'"memory_database":"postgres"'* ]]
[[ "$health" == *'"short_term_backend":"postgres-checkpointer"'* ]]
[[ "$health" == *'"prediction":"enabled"'* ]]

echo "[check] verifying Web login route"
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:3003/login >/dev/null

echo "[check] X1C private deployment is healthy"
