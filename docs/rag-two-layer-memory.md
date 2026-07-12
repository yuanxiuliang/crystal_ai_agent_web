# RAG Two-Layer Memory

The production profile runs on the ThinkPad Ubuntu host. Public traffic reaches the
host through the low-resource gateway and a private reverse tunnel. The gateway never
stores RAG corpus data, conversation state, or user memory.

## Boundaries

```text
Short-term memory: session_id, LangGraph PostgreSQL checkpointer
Long-term memory: user_id only, PostgreSQL + JSONB + pgvector
Knowledge corpus: Milvus, separate from user memory
```

`project_id` and `laboratory_id` are deliberately not part of the memory schema. A user
can still store research summaries and experimental constraints, but every long-term
record is owned and isolated solely by `user_id`.

## Short-Term Memory

The `checkpoint_sessions` mapping is keyed only by `user_id + session_id`. Its initial
opaque LangGraph thread id derives from those two values, preventing users who choose
the same session label from sharing a thread. The persisted state contains only a
bounded message window, conversation summary, active context, and short-memory slots.

After `MEMORY_THREAD_ROLLOVER_TURNS` completed turns (100 by default), the service
copies that compact state into a new LangGraph thread, atomically switches the mapping,
and then uses `AsyncPostgresSaver.adelete_thread()` to delete the old thread. It never
deletes individual checkpoint rows. The `rag-memory-worker` also removes threads whose
session mapping has expired, so idle sessions cannot accumulate indefinitely.

At the end of every turn, the graph removes retrieval records, evidence packs, answer
payloads, citations, trace entries, and other turn-only fields before the next
checkpoint is saved. Chat history for user-facing display belongs in a separate product
table and is not loaded into the graph state indefinitely.

## Long-Term Memory

`memory_items` uses a user-scoped identity key:

```text
(user_id, memory_type, memory_key)
```

Supported types are:

```text
preference
constraint
research_profile
project_digest
confirmed_fact
```

`constraint` and `preference` are read deterministically. Other relevant items are
ranked using structured query terms plus the pgvector cosine similarity of their
MiniLM 384-dimensional embeddings. Each answer receives at most the configured number
of long-term records and characters.

Explicit user requests are written synchronously. Every created or updated item enqueues
an `embed_memory` job in PostgreSQL. The separate `rag-memory-worker` embeds the compact
memory record and writes the vector later, so user-facing response latency is unchanged.

## Retention

```text
Short-term message window:        10 messages by default
Conversation summary:             bounded by MEMORY_SUMMARY_MAX_CHARS
Checkpoint thread rollover:       100 completed turns by default
Active long-term memories/user:   200 by default
Long-term prompt injection:       8 items / 1,800 characters by default
Session state expiry:             30 days by default
Memory event expiry:              90 days by default
```

Long-term entries are updated by identity key instead of appended indefinitely. A user
memory management API can later expose the same `memory_items` records for inspection,
editing, and deletion.
