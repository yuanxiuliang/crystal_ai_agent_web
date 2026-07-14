#!/usr/bin/env bash
set -Eeuo pipefail

# One-time Ubuntu host preparation for the ThinkPad RAG deployment.
# Run on the X1C as: sudo bash ~/bootstrap-x1c-rag.sh yuanx

DEPLOY_USER="${1:-}"
MIRROR_BASE="https://mirrors.aliyun.com/ubuntu/"
SOURCES_FILE="/etc/apt/sources.list.d/ubuntu.sources"
DOCKER_CONFIG="/etc/docker/daemon.json"
BACKUP_DIR="/var/backups/agentweb-rag"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

if [[ -z "${DEPLOY_USER}" ]] || ! id "${DEPLOY_USER}" >/dev/null 2>&1; then
  echo "Usage: sudo bash $0 <existing-deploy-user>" >&2
  exit 2
fi

if [[ ! -f "${SOURCES_FILE}" ]]; then
  echo "Expected Ubuntu source file is missing: ${SOURCES_FILE}" >&2
  exit 1
fi

wait_for_apt_lock() {
  local elapsed=0
  local timeout_seconds=600
  local locks=(
    /var/lib/dpkg/lock-frontend
    /var/lib/dpkg/lock
    /var/lib/apt/lists/lock
    /var/cache/apt/archives/lock
  )

  while fuser "${locks[@]}" >/dev/null 2>&1; do
    if (( elapsed >= timeout_seconds )); then
      echo "Timed out waiting for Ubuntu package maintenance to finish." >&2
      exit 1
    fi
    echo "Waiting for Ubuntu package maintenance to release its lock..."
    sleep 10
    ((elapsed += 10))
  done
}

install -d -m 0755 "${BACKUP_DIR}"
shopt -s nullglob
for stale_backup in /etc/apt/sources.list.d/ubuntu.sources.before-agentweb-*; do
  mv "${stale_backup}" "${BACKUP_DIR}/"
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
sources_backup="${BACKUP_DIR}/ubuntu.sources.before-agentweb-${timestamp}"
cp --preserve=mode,ownership,timestamps "${SOURCES_FILE}" "${sources_backup}"

sed -i \
  -e "s|http://archive.ubuntu.com/ubuntu/|${MIRROR_BASE}|g" \
  -e "s|http://security.ubuntu.com/ubuntu/|${MIRROR_BASE}|g" \
  "${SOURCES_FILE}"

wait_for_apt_lock
if ! apt-get update; then
  cp --preserve=mode,ownership,timestamps "${sources_backup}" "${SOURCES_FILE}"
  wait_for_apt_lock
  apt-get update
  echo "Domestic Ubuntu mirror update failed; restored the previous source file." >&2
  exit 1
fi

wait_for_apt_lock
DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
  ca-certificates \
  curl \
  git \
  rsync \
  python3-pip \
  python3-venv \
  build-essential \
  docker.io \
  docker-compose-v2

install -d -m 0755 /etc/docker
if [[ ! -e "${DOCKER_CONFIG}" ]]; then
  cat >"${DOCKER_CONFIG}" <<'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io"
  ]
}
EOF
else
  echo "Preserving existing Docker configuration: ${DOCKER_CONFIG}"
fi

systemctl enable --now docker.service
systemctl restart docker.service
usermod -aG docker "${DEPLOY_USER}"

docker version --format 'Docker {{.Server.Version}} is ready.'
docker compose version
echo "Bootstrap complete. Reconnect as ${DEPLOY_USER} before using Docker without sudo."
