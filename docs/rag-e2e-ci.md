# RAG End-to-End CI Contract

The `rag-e2e` GitHub Actions job starts a disposable RAG stack. It uses an isolated PostgreSQL
database, Milvus, MinIO, etcd, a five-record MiniLM corpus, the production API/Web Dockerfiles,
and the vendored prediction model. It never mounts X1C volumes or production user data.

The job requires three GitHub Actions secrets:

- `RAG_E2E_LLM_BASE_URL`: OpenAI-compatible base URL without `/chat/completions`.
- `RAG_E2E_LLM_API_KEY`: credential for the real LLM endpoint.
- `RAG_E2E_LLM_MODEL`: exact production-equivalent model name.

The API verifies `llm_backend=openai-compatible` before running. A missing secret or unavailable
LLM fails the protected `main` CI run and therefore prevents X1C promotion.

The test contract covers:

1. first-login registration, invalid-password rejection, and authenticated sessions;
2. real MiniLM (384 dimensions) plus Milvus retrieval of an exact TaAs record;
3. same-session TaAs follow-up resolution through the PostgreSQL LangGraph checkpointer;
4. Mn3GaN retrieval insufficiency, rejection of Mn3Ge as direct evidence, and real predictor
   fallback with an explicit unverified label;
5. evidence-only Mn3GaN requests, which must not invoke prediction or fabricate a DOI;
6. user-scoped conversation access and PostgreSQL long-memory isolation;
7. browser rendering of the real-evidence and prediction answer variants;
8. a second bootstrap run, which must reuse the complete test collection rather than re-embed it.
9. editing an earlier user question, which must remove the later conversation branch, reset the
   LangGraph short-term checkpoint, and regenerate from the retained earlier context only.

The test uses synthetic `10.5555/e2e.*` DOI values and a unique `growth_records_e2e` collection.
It is intentionally separate from the 6,292-record production corpus.
Its host ports default to `18003` for the API and `13003` for the Web UI so local development
services on `8003` and `3003` remain untouched.

Run it locally only with a real LLM endpoint in the environment:

```bash
export RAG_E2E_LLM_BASE_URL="https://example.invalid/v1"
export RAG_E2E_LLM_API_KEY="..."
export RAG_E2E_LLM_MODEL="..."
pnpm install --frozen-lockfile
pnpm exec playwright install chromium
./e2e/run-rag-e2e.sh
```

`RAG_E2E_IMAGE_PROXY` is optional. It lets a domestic developer network or self-hosted runner
pull all disposable-stack images through a registry proxy without changing the production
Dockerfiles. GitHub-hosted runners must leave it unset and pull the official base images directly;
registry proxies can reject GitHub Runner traffic or return incomplete image layers. For a domestic
developer network, use a currently reachable proxy such as:

```bash
export RAG_E2E_IMAGE_PROXY=docker.1ms.run
```
