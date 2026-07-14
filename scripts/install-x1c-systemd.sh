#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_TEMPLATE="$ROOT_DIR/infra/systemd/agentweb-rag.service"
DEPLOY_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo ./scripts/install-x1c-systemd.sh" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/infra/compose/.env.x1c" ]]; then
  echo "Create infra/compose/.env.x1c and validate deployment before enabling systemd." >&2
  exit 1
fi

if [[ ! -f "$UNIT_TEMPLATE" ]]; then
  echo "Missing systemd template: $UNIT_TEMPLATE" >&2
  exit 1
fi

escaped_root="$(printf '%s' "$ROOT_DIR" | sed 's/[&|]/\\&/g')"
escaped_user="$(printf '%s' "$DEPLOY_USER" | sed 's/[&|]/\\&/g')"

sed \
  -e "s|__DEPLOY_ROOT__|${escaped_root}|g" \
  -e "s|__DEPLOY_USER__|${escaped_user}|g" \
  "$UNIT_TEMPLATE" >/etc/systemd/system/agentweb-rag.service

chmod 0644 /etc/systemd/system/agentweb-rag.service
systemctl daemon-reload
systemctl enable agentweb-rag.service
systemctl start agentweb-rag.service
systemctl --no-pager --full status agentweb-rag.service
