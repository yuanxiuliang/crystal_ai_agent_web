#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAG_UNIT_TEMPLATE="$ROOT_DIR/infra/systemd/agentweb-rag.service"
TUNNEL_UNIT_TEMPLATE="$ROOT_DIR/infra/systemd/agentweb-rag-tunnel.service"
DEPLOY_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo ./scripts/install-x1c-systemd.sh" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/infra/compose/.env.x1c" ]]; then
  echo "Create infra/compose/.env.x1c and validate deployment before enabling systemd." >&2
  exit 1
fi

for template in "$RAG_UNIT_TEMPLATE" "$TUNNEL_UNIT_TEMPLATE"; do
  if [[ ! -f "$template" ]]; then
    echo "Missing systemd template: $template" >&2
    exit 1
  fi
done

DEPLOY_HOME="$(getent passwd "$DEPLOY_USER" | cut -d: -f6)"
if [[ -z "$DEPLOY_HOME" || ! -d "$DEPLOY_HOME" ]]; then
  echo "Cannot determine the home directory for deployment user: $DEPLOY_USER" >&2
  exit 1
fi

for required_file in \
  "$DEPLOY_HOME/.ssh/id_ed25519_agentweb_tunnel" \
  "$DEPLOY_HOME/.ssh/known_hosts"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing reverse-tunnel prerequisite: $required_file" >&2
    exit 1
  fi
done

escaped_root="$(printf '%s' "$ROOT_DIR" | sed 's/[&|]/\\&/g')"
escaped_user="$(printf '%s' "$DEPLOY_USER" | sed 's/[&|]/\\&/g')"
escaped_home="$(printf '%s' "$DEPLOY_HOME" | sed 's/[&|]/\\&/g')"

sed \
  -e "s|__DEPLOY_ROOT__|${escaped_root}|g" \
  -e "s|__DEPLOY_USER__|${escaped_user}|g" \
  "$RAG_UNIT_TEMPLATE" >/etc/systemd/system/agentweb-rag.service

sed \
  -e "s|__DEPLOY_USER__|${escaped_user}|g" \
  -e "s|__DEPLOY_HOME__|${escaped_home}|g" \
  "$TUNNEL_UNIT_TEMPLATE" >/etc/systemd/system/agentweb-rag-tunnel.service

chmod 0644 \
  /etc/systemd/system/agentweb-rag.service \
  /etc/systemd/system/agentweb-rag-tunnel.service
systemctl daemon-reload
systemctl enable agentweb-rag.service agentweb-rag-tunnel.service
systemctl start agentweb-rag.service
systemctl restart agentweb-rag-tunnel.service
systemctl --no-pager --full status agentweb-rag.service
systemctl --no-pager --full status agentweb-rag-tunnel.service
