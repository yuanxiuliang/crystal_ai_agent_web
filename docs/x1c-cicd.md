# X1C CI/CD

## Delivery Contract

The delivery path is deliberately split into two gates:

```text
developer commit and push to main
  -> GitHub Actions CI
  -> X1C poller sees the same commit with CI=success
  -> isolated candidate container tests on X1C
  -> production Compose deployment
```

GitHub CI runs backend lint/tests, RAG Web type checking/building, deployment shell/asset checks,
and a disposable real-LLM RAG end-to-end stack. The end-to-end job starts isolated PostgreSQL,
Milvus, MiniLM, prediction, API, and Web services with a five-record synthetic corpus. It checks
authentication, user isolation, PostgreSQL-backed short and long memory, exact-record retrieval,
prediction fallback, evidence-only limits, SSE contracts, browser rendering, and idempotent
bootstrap. The production Compose definition is then validated by the X1C candidate promotion path
before any production container is changed. The workflow file is `.github/workflows/ci.yml`; its
exact workflow identity is part of the X1C deployment contract.

The GitHub repository must define `RAG_E2E_LLM_BASE_URL`, `RAG_E2E_LLM_API_KEY`, and
`RAG_E2E_LLM_MODEL` as Actions secrets. They are injected only into the `main` push or manually
triggered E2E job, never into pull-request code. A missing or unavailable real LLM makes that
workflow fail, so the X1C timer cannot promote the revision. Details are in `docs/rag-e2e-ci.md`.

The X1C poller never deploys a commit that lacks a successful GitHub CI run. It then creates a
detached worktree under `/home/yuanx/agentweb-rag-releases/<commit>`, builds a candidate API image
from that worktree, and runs backend checks in a container with all of the following boundaries:

- no Docker network access;
- read-only root filesystem;
- temporary `/tmp` only;
- temporary SQLite memory and prediction databases;
- no connection to production PostgreSQL, Milvus, MinIO, or etcd.

Passing the candidate test is required before the Git-managed production checkout is advanced and
the normal `scripts/deploy-x1c-rag.sh` deployment runs. Persistent production volumes are never
part of candidate testing.

The X1C checkout is release-only. It advances by checking out the exact tested commit in detached
HEAD mode, which also works with the shallow first clone. The updater refuses any tracked local
change before doing so; application configuration remains in ignored files outside Git history.

## Public Relay Cache Rule

The public relay configuration is versioned at
`infra/relay/nginx/aicrystal.yuanxiuliang.cn.conf`. It must explicitly disable `proxy_cache` for
both `/api/` and `/`. The relay Nginx installation enables a global proxy-cache zone for unrelated
sites; leaving the web `location /` block unqualified can otherwise retain a prior Next HTML
response and its old JavaScript asset references after a successful X1C deployment.

Install or update the versioned virtual host as root, then validate and reload Nginx:

```bash
install -m 0644 infra/relay/nginx/aicrystal.yuanxiuliang.cn.conf \
  /www/server/panel/vhost/nginx/aicrystal.yuanxiuliang.cn.conf
nginx -t
systemctl reload nginx
```

## First Bootstrap

The first CI/CD-enabled commit must already be present on GitHub. On the X1C, run the committed
bootstrap script as the normal deployment user:

```bash
cd /home/yuanx/agentweb-rag
chmod +x scripts/bootstrap-x1c-cd.sh
./scripts/bootstrap-x1c-cd.sh
```

The bootstrap script does the following atomically enough for this small single-host deployment:

1. Shallow-clones `main` into a Git-managed replacement directory. This keeps the first transfer
   bounded while retaining normal Git fetch support for future updates.
2. Copies the existing ignored `infra/compose/.env.x1c` deployment configuration with mode `0600`.
3. Preserves the old copied source directory as a timestamped backup.
4. Reinstalls the existing production start unit against the Git-managed path.
5. Installs and enables the five-minute CD timer.
6. Starts one deployment check immediately.

The repository is public at the time this document was written, so X1C uses read-only HTTPS Git
and GitHub Actions metadata without any token. If it becomes private, use a read-only Git deploy
key for the clone/fetch path and add a fine-grained, read-only Actions token as
`GITHUB_API_TOKEN` in `/etc/agentweb-rag-cd.env`. Do not put credentials in the repository or in
`infra/compose/.env.x1c`.

The CD environment file is intentionally root-owned and mode `0600`. systemd reads it before
starting the unprivileged deployment service and passes only its variables to that service; the
deployment script does not need direct filesystem read access to the file.

## Operations

View recent deployment decisions:

```bash
journalctl -u agentweb-rag-cd.service -n 100 --no-pager
systemctl status agentweb-rag-cd.timer
```

Run a check immediately rather than waiting for the next poll:

```bash
sudo systemctl start agentweb-rag-cd.service
```

A failed candidate creates `.cd-state/failed-<commit>` in the deployment checkout and preserves
its detached worktree for diagnosis. The timer deliberately does not repeatedly rebuild a known
failing commit. After correcting the issue, push a new commit. To intentionally retry the same
commit, remove only its matching failed marker and start the CD service again.

## Resource Boundary

The full disposable Milvus/PostgreSQL E2E stack runs in GitHub Actions, never on the X1C. The X1C
candidate gate remains a network-disabled API container and does not re-embed the 6,292-record
production corpus. This preserves the X1C's CPU and memory budget while making the GitHub CI
result the required real-LLM release gate.
