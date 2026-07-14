#!/usr/bin/env bash
set -Eeuo pipefail

CD_ENV_FILE="${AGENTWEB_CD_ENV_FILE:-/etc/agentweb-rag-cd.env}"
if [[ -f "$CD_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CD_ENV_FILE"
fi

DEPLOY_ROOT="${AGENTWEB_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BRANCH="${AGENTWEB_BRANCH:-main}"
REPOSITORY="${AGENTWEB_REPOSITORY:-}"
RELEASES_ROOT="${AGENTWEB_RELEASES_ROOT:-${DEPLOY_ROOT}-releases}"
STATE_ROOT="$DEPLOY_ROOT/.cd-state"
LOCK_FILE="$STATE_ROOT/deploy.lock"

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "[cd] missing required configuration: $name" >&2
    exit 2
  fi
}

require_value "AGENTWEB_REPOSITORY" "$REPOSITORY"
command -v curl >/dev/null
command -v docker >/dev/null
command -v git >/dev/null
command -v jq >/dev/null
command -v flock >/dev/null

mkdir -p "$STATE_ROOT" "$RELEASES_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[cd] another deployment check is already running; skipping"
  exit 0
fi

cd "$DEPLOY_ROOT"
if [[ ! -d .git ]]; then
  echo "[cd] deployment root is not a Git checkout: $DEPLOY_ROOT" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[cd] deployment checkout has tracked local changes; refusing automatic update" >&2
  exit 1
fi

remote_sha="$(git ls-remote --refs origin "refs/heads/$BRANCH" | awk 'NR == 1 { print $1 }')"
if [[ ! "$remote_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[cd] unable to resolve origin/$BRANCH" >&2
  exit 1
fi

deployed_sha="$(cat "$STATE_ROOT/deployed-sha" 2>/dev/null || true)"
if [[ "$remote_sha" == "$deployed_sha" ]]; then
  echo "[cd] production already runs $remote_sha"
  exit 0
fi

failed_marker="$STATE_ROOT/failed-$remote_sha"
if [[ -f "$failed_marker" ]]; then
  echo "[cd] candidate $remote_sha previously failed; waiting for a newer commit"
  exit 0
fi

github_headers=(-H "Accept: application/vnd.github+json")
if [[ -n "${GITHUB_API_TOKEN:-}" ]]; then
  github_headers+=(-H "Authorization: Bearer $GITHUB_API_TOKEN")
fi

workflow_json="$(curl --fail --silent --show-error --max-time 30 \
  "${github_headers[@]}" \
  "https://api.github.com/repos/$REPOSITORY/actions/workflows/ci.yml/runs?head_sha=$remote_sha&per_page=20")"
ci_status="$(printf '%s' "$workflow_json" | jq -r --arg sha "$remote_sha" '
  [ .workflow_runs[]
    | select(.head_sha == $sha and .event == "push")
    | {id, status, conclusion}
  ] | sort_by(.id) | last // {status: "missing", conclusion: ""}
  | "\(.status) \(.conclusion // \"\")"
')"

case "$ci_status" in
  "completed success")
    echo "[cd] GitHub CI passed for $remote_sha"
    ;;
  "completed "*)
    echo "[cd] GitHub CI did not pass for $remote_sha: $ci_status"
    exit 0
    ;;
  *)
    echo "[cd] GitHub CI is not ready for $remote_sha: $ci_status"
    exit 0
    ;;
esac

git fetch --prune origin "$BRANCH"
git cat-file -e "$remote_sha^{commit}"

release_root="$RELEASES_ROOT/$remote_sha"
if [[ -e "$release_root" ]]; then
  candidate_sha="$(git -C "$release_root" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$candidate_sha" != "$remote_sha" ]]; then
    echo "[cd] release worktree path is occupied by a different revision: $release_root" >&2
    exit 1
  fi
else
  git worktree add --detach "$release_root" "$remote_sha"
fi

if [[ ! -f "$release_root/infra/compose/.env.x1c" ]]; then
  install -D -m 600 "$DEPLOY_ROOT/infra/compose/.env.x1c" \
    "$release_root/infra/compose/.env.x1c"
fi

short_sha="${remote_sha:0:12}"
candidate_api_image="agentweb-rag-api:candidate-$short_sha"
candidate_test_image="agentweb-rag-api-test:candidate-$short_sha"

run_candidate_tests() {
  echo "[cd] building isolated API candidate $remote_sha"
  docker build --tag "$candidate_api_image" \
    --file "$release_root/infra/docker/rag-api.Dockerfile" "$release_root"

  docker build --build-arg "BASE_IMAGE=$candidate_api_image" \
    --tag "$candidate_test_image" \
    --file "$release_root/infra/docker/rag-api-test.Dockerfile" "$release_root"

  echo "[cd] running backend checks in an isolated, network-disabled test container"
  docker run --rm --network none --read-only \
    --tmpfs /tmp:rw,mode=1777,size=512m \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e XDG_CACHE_HOME=/tmp/cache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -e MEMORY_DATABASE_URL=sqlite:////tmp/rag-memory.sqlite3 \
    -e PREDICTION_DATABASE_URL=sqlite:////tmp/rag-prediction.sqlite3 \
    -e MEMORY_CHECKPOINT_BACKEND=none \
    -e MEMORY_SEMANTIC_SEARCH_ENABLED=false \
    -e PREDICTION_DEVICE=cpu \
    -e PREDICTION_TORCH_THREADS=2 \
    -e PREDICTION_MODEL_DIR=/opt/agentweb/services/rag-api/models/growth-route-transformer/v2.0.0 \
    -e PYTHONPATH=/opt/agentweb/services/rag-api \
    "$candidate_test_image" \
    sh -eu -c 'ruff check --no-cache src tests && pytest -q -p no:cacheprovider tests'
}

if ! run_candidate_tests; then
  touch "$failed_marker"
  echo "[cd] candidate test failed; production remains on ${deployed_sha:-the existing release}" >&2
  exit 1
fi

echo "[cd] candidate passed; advancing production checkout to $remote_sha"
git merge --ff-only "$remote_sha"

echo "[cd] rebuilding and deploying the tested production revision"
RAG_DEPLOY_PULL_BASE_IMAGES=0 "$DEPLOY_ROOT/scripts/deploy-x1c-rag.sh"

printf '%s\n' "$remote_sha" >"$STATE_ROOT/deployed-sha"
rm -f "$failed_marker"
docker image rm "$candidate_test_image" "$candidate_api_image" >/dev/null 2>&1 || true
echo "[cd] deployed $remote_sha successfully"
