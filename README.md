# AgentWeb

AgentWeb is a unified single-crystal-growth research RAG workbench. One authenticated Web
application combines RAG dialogue, literature and experimental-record retrieval, and
formula-conditioned growth-route prediction.

```text
Web application: apps/rag-platform
Business API:    services/rag-api
Corpus:          Milvus hybrid retrieval
User state:      PostgreSQL + pgvector + LangGraph checkpointer
Prediction:      local Growth Route Transformer bundle
```

The former search and inference applications are legacy placeholders and are scheduled for
workspace cleanup. The active architecture and hard rules are defined in
`docs/project-constraints.md`; the capability design is in
`docs/unified-research-platform-design.md`.

## One-command RAG development startup

The default development startup starts local PostgreSQL + pgvector, Milvus, the bounded
memory worker, waits for dependencies to become ready, imports the real growth records into
the MiniLM-backed collection when needed, and then opens the interactive RAG CLI:

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

Start the unified LangGraph research assistant from the repository root:

```bash
scripts/growth-rag-chat
```

Ask one question and exit:

```bash
scripts/growth-rag-chat "ZnIn2S4 的 CVT 生长温度是多少？"
```

The CLI defaults to trace output and `top_k=3` for development-stage validation. Use
`/trace off` inside interactive mode to hide node events.

## Direct Prediction CLI

The formula-conditioned route predictor is available independently of the Chat graph. It
validates the bundled checkpoint digest before first use, then reuses one CPU model instance
for later requests in the same process.

```bash
cd services/rag-api
.venv/bin/rag-predict --formula Mn3GaN --user-id researcher-alice
```

The response contains up to three `Flux` or `CVT` candidate routes, the model version and
artifact digest, temperature/duration bin ranges, and explicit validation warnings. Route
ranking weights are only relative ordering values, not experimental success probabilities.

When launched through `./scripts/dev-rag.sh`, prediction runs use the same PostgreSQL database
as the RAG memory profile. Each run is owned by its `user_id` and can be inspected through:

```text
POST /api/rag/predict
GET  /api/rag/prediction-runs
```

The public Web API derives `user_id` from an authenticated HttpOnly-cookie session; it does not
accept a browser-supplied `user_id`. Development CLI commands retain explicit `--user-id` and
`--session-id` switches only for local memory-isolation tests.

## Web Accounts And Sessions

The Web application opens at `http://localhost:3003/login`. An email address is the unique
account identifier. The first successful login with an unregistered email and a valid password
creates an account; a later login with the same email verifies its stored Argon2id password hash.
This product intentionally does not verify email ownership or send an email code.

After login, the account receives an HttpOnly session cookie. Chat sessions, visible message
history, long-term memory, and prediction-run history are all scoped to that server-side account
identity. The sidebar creates, renames, lists, and deletes only the current account's sessions.

## Chat Evidence Selection

Chat is retrieval-first. It uses sufficient, traceable literature or experimental records as
the only evidence source and does not load the prediction model on that path. When retrieval
finishes normally but returns no usable evidence, Chat may run the route predictor for a
candidate-route request with one unambiguous formula. This includes direct operational questions
such as "Mn3GaN怎么做", "我要长 Mn3GaN 单晶", "如何制备 Mn3GaN", and a follow-up
request to predict after the formula has become the active context. After completed insufficient
retrieval, LangGraph invokes the model automatically; it does not require a literal "推荐方案".
Prediction output is labeled as model-generated and contains no literature citation.

The direct retrieval endpoint, `POST /api/rag/retrieve`, is evidence-only. It never turns an
empty or insufficient retrieval result into a prediction. A Milvus outage also never triggers
prediction because service unavailability cannot establish that the corpus lacks real data.

## X1C Production Deployment

The formal deployment profile runs the complete RAG service on the X1C Ubuntu host. It uses a
CPU-only Python 3.12 API container, PostgreSQL + pgvector, Milvus, the pre-exported MiniLM INT8
embedding model, the local growth-route predictor, and a separate memory worker. PostgreSQL,
Milvus, MinIO, and etcd have no host port mappings. Web and API diagnostics bind only to
`127.0.0.1` on the X1C.

The production image pins `torch==2.6.0+cpu`; it never resolves generic Linux PyTorch or CUDA
dependencies. Docker, pnpm, pip, and the PyTorch wheel install all use domestic mirrors.

The complete private-deployment procedure is in
[`docs/x1c-private-deployment.md`](docs/x1c-private-deployment.md). Do not configure the
constrained public server until this X1C stack has passed private auth, retrieval, prediction,
and memory-isolation checks.
