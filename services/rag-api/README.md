# RAG API

FastAPI service for the AgentWeb RAG dialogue platform.

For a one-command local startup of PostgreSQL + pgvector, Milvus, memory initialization,
the long-term-memory worker, data ingestion, and the interactive RAG CLI, run this from
the repository root:

```bash
./scripts/dev-rag.sh
```

Use `./scripts/dev-rag.sh --rebuild-milvus` only when the Milvus collection needs to
be recreated, such as after changing the embedding model or vector dimension.
The one-command startup also detects the previous 1024-dimensional collection and
refreshes it automatically for the current 384-dimensional MiniLM embedding model.

Use `./scripts/dev-rag.sh --web` to start the FastAPI backend and Web UI instead.

The script uses the local PostgreSQL memory profile by default:

```text
long-term memory: PostgreSQL + JSONB + pgvector, scoped only by user_id
short-term memory: LangGraph PostgreSQL checkpointer, scoped by user_id + session_id
```

For the intentionally limited SQLite fallback, use:

```bash
RAG_MEMORY_PROFILE=sqlite ./scripts/dev-rag.sh
```

For local multi-user memory tests, the CLI accepts an explicit development identity. These
arguments are rejected in `--web` mode and must not be treated as production authentication:

```bash
./scripts/dev-rag.sh --user-id researcher-alice --session-id alice-session-1
```

Inside the development CLI, `/whoami` prints the selected identity and `/memory` lists
only the long-term records visible to that user. They are diagnostics, not production
account-management commands.

The current development phase validates the RAG agent workflow from the command line. The Web frontend is kept in the repository but is not the active validation entrypoint.

## Memory Persistence

The direct API fallback is a bounded local SQLite store at
`data/runtime/rag-memory.sqlite3`. It persists only the latest short-term window,
fixed-size conversation summary, and a capped set of structured long-term memories. It
does not embed memory entries or start another resident service.

The default budgets are intentionally small for a 1 vCPU / 1 GiB host:

```text
recent messages per session: 10
conversation summary:       1,200 characters
active long memories/user:  200
memories injected/turn:     8 / 1,800 characters
inactive session state:     expires after 30 days
```

For a multi-instance production deployment, configure `MEMORY_DATABASE_URL` with an
external `postgresql://...` URL and install the PostgreSQL extra. Do not co-locate a
PostgreSQL container with Milvus on the public 1 GiB gateway.

```bash
cd services/rag-api
.venv/bin/pip install -e '.[postgres]'
```

For the ThinkPad deployment profile, PostgreSQL and pgvector run beside Milvus on the
local Ubuntu host. The public 1 GiB gateway does not run either service. Configure:

```bash
MEMORY_DATABASE_URL=postgresql://agentweb:agentweb@127.0.0.1:5432/agentweb
MEMORY_CHECKPOINT_BACKEND=postgres
MEMORY_SEMANTIC_SEARCH_ENABLED=true
```

Initialize database schemas and the LangGraph checkpointer before starting the API:

```bash
.venv/bin/python -m src.cli.rag_memory_init
```

Run the bounded long-term-memory worker as a separate local process. It only embeds
confirmed memory entries and never embeds raw chat history:

```bash
.venv/bin/python -m src.cli.rag_memory_worker
```

CLI entrypoint:

```bash
cd services/rag-api
source ~/.zshrc
.venv/bin/python -m src.cli.rag_chat --trace
```

Single-question mode:

```bash
.venv/bin/python -m src.cli.rag_chat "ZnIn2S4 的 CVT 生长温度是多少？" --trace
```

Offline workflow check with mock LLM:

```bash
.venv/bin/python -m src.cli.rag_chat "ZnIn2S4 的 CVT 生长温度是多少？" --trace --mock
```

Inspect current RAG configuration:

```bash
.venv/bin/python -m src.cli.rag_config
```

Milvus is the target retrieval backend for the next phase: BM25 sparse search + dense vector search + RRF fusion. Start the local Milvus stack from the repository root:

```bash
docker compose -f infra/compose/docker-compose.dev.yml up -d milvus-etcd milvus-minio milvus-standalone
```

Then check connectivity:

```bash
cd services/rag-api
.venv/bin/python -m src.cli.rag_config --check-milvus
```

FastAPI service entrypoint, retained for later Web/API integration:

```text
POST /api/rag/chat/stream
```
