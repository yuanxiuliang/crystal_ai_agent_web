#!/usr/bin/env bash
set -Eeuo pipefail

# One-time migration from a copied source tree to a Git-managed X1C deployment checkout.
# Run as the deployment user after the first CI-enabled commit has been pushed to GitHub.
REPOSITORY_URL="${AGENTWEB_REPOSITORY_URL:-https://github.com/yuanxiuliang/crystal_ai_agent_web.git}"
BRANCH="${AGENTWEB_BRANCH:-main}"
DEPLOY_ROOT="${AGENTWEB_DEPLOY_ROOT:-$HOME/agentweb-rag}"
CANDIDATE_ROOT="${AGENTWEB_CANDIDATE_ROOT:-${DEPLOY_ROOT}.git-candidate}"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this bootstrap as the normal deployment user, not root." >&2
  exit 1
fi

if [[ ! -d "$DEPLOY_ROOT" ]]; then
  echo "Missing current deployment root: $DEPLOY_ROOT" >&2
  exit 1
fi

if [[ ! -f "$DEPLOY_ROOT/infra/compose/.env.x1c" ]]; then
  echo "Missing existing deployment environment: $DEPLOY_ROOT/infra/compose/.env.x1c" >&2
  exit 1
fi

if [[ ! -d "$DEPLOY_ROOT/.git" ]]; then
  backup_root="${DEPLOY_ROOT}.pre-git-$(date +%Y%m%dT%H%M%SZ)"
  if [[ -d "$CANDIDATE_ROOT/.git" ]]; then
    echo "[bootstrap-cd] reusing existing Git candidate: $CANDIDATE_ROOT"
    git -C "$CANDIDATE_ROOT" -c http.version=HTTP/1.1 fetch --depth 1 origin "$BRANCH"
    git -C "$CANDIDATE_ROOT" checkout -B "$BRANCH" FETCH_HEAD
  elif [[ -e "$CANDIDATE_ROOT" ]]; then
    echo "A previous Git candidate directory still exists: $CANDIDATE_ROOT" >&2
    echo "Inspect it before rerunning this bootstrap; it will not be deleted automatically." >&2
    exit 1
  else
    echo "[bootstrap-cd] shallow-cloning $REPOSITORY_URL ($BRANCH)"
    git -c http.version=HTTP/1.1 clone --depth 1 --branch "$BRANCH" --single-branch \
      "$REPOSITORY_URL" "$CANDIDATE_ROOT"
  fi
  install -D -m 600 "$DEPLOY_ROOT/infra/compose/.env.x1c" \
    "$CANDIDATE_ROOT/infra/compose/.env.x1c"
  mv "$DEPLOY_ROOT" "$backup_root"
  mv "$CANDIDATE_ROOT" "$DEPLOY_ROOT"
  echo "[bootstrap-cd] preserved prior copied source at $backup_root"
fi

cd "$DEPLOY_ROOT"
git config pull.ff only
git config http.version HTTP/1.1
sudo ./scripts/install-x1c-systemd.sh
sudo ./scripts/install-x1c-cd.sh
sudo systemctl start agentweb-rag-cd.service
echo "[bootstrap-cd] Git-managed CD is enabled; inspect progress with:"
echo "journalctl -u agentweb-rag-cd.service -n 100 --no-pager"
