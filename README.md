# AgentWeb

AgentWeb is a monorepo for three isolated AI web projects:

1. Search platform: `apps/search-platform`
2. Model inference platform: `apps/inference-platform`
3. RAG dialogue platform: `apps/rag-platform`

Phase 1 focuses on the RAG dialogue platform and `services/rag-api`.

## One-command RAG development startup

The default development startup starts Milvus, waits for it to become ready,
imports the real growth records into the MiniLM-backed collection when needed,
and then opens the interactive RAG CLI:

```bash
./scripts/dev-rag.sh
```

The first run imports the 384-dimensional `all-MiniLM-L6-v2` embeddings. Later runs
skip the import when the collection is ready. Web frontend dependencies are installed
only when using `--web`.
If an old collection uses a different vector dimension or contains incomplete data,
the startup check automatically recreates it for the current MiniLM configuration.

To start the FastAPI backend and Web UI instead:

```bash
./scripts/dev-rag.sh --web
```

Open `http://localhost:3003/chat` after the script reports that the frontend is ready.

To rebuild the collection explicitly, for example after changing embedding dimensions:

```bash
./scripts/dev-rag.sh --rebuild-milvus
```

Core constraints are documented in `docs/project-constraints.md`.

## Growth RAG CLI

Start the project 3 LangGraph dialogue app from the repository root:

```bash
scripts/growth-rag-chat
```

Ask one question and exit:

```bash
scripts/growth-rag-chat "ZnIn2S4 的 CVT 生长温度是多少？"
```

The CLI defaults to trace output and `top_k=3` for development-stage validation. Use
`/trace off` inside interactive mode to hide node events.
