#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_TEMPLATE="$ROOT_DIR/infra/systemd/agentweb-rag-cd.service"
TIMER_TEMPLATE="$ROOT_DIR/infra/systemd/agentweb-rag-cd.timer"
ENV_TEMPLATE="$ROOT_DIR/infra/cd/agentweb-rag-cd.env.example"
DEPLOY_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo ./scripts/install-x1c-cd.sh" >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/.git" ]]; then
  echo "The CD installer requires a Git checkout: $ROOT_DIR" >&2
  exit 1
fi

for file in "$SERVICE_TEMPLATE" "$TIMER_TEMPLATE" "$ENV_TEMPLATE"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing CD installation asset: $file" >&2
    exit 1
  fi
done

if [[ ! -f /etc/agentweb-rag-cd.env ]]; then
  install -m 600 "$ENV_TEMPLATE" /etc/agentweb-rag-cd.env
fi

escaped_root="$(printf '%s' "$ROOT_DIR" | sed 's/[&|]/\\&/g')"
escaped_user="$(printf '%s' "$DEPLOY_USER" | sed 's/[&|]/\\&/g')"

sed \
  -e "s|__DEPLOY_ROOT__|${escaped_root}|g" \
  -e "s|__DEPLOY_USER__|${escaped_user}|g" \
  "$SERVICE_TEMPLATE" >/etc/systemd/system/agentweb-rag-cd.service
install -m 0644 "$TIMER_TEMPLATE" /etc/systemd/system/agentweb-rag-cd.timer

chmod 0755 "$ROOT_DIR/scripts/x1c-auto-deploy.sh"
systemctl daemon-reload
systemctl enable agentweb-rag-cd.timer
systemctl start agentweb-rag-cd.timer
systemctl --no-pager --full status agentweb-rag-cd.timer
