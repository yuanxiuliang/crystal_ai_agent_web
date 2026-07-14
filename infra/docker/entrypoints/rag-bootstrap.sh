#!/bin/sh
set -eu

attempts="${RAG_BOOTSTRAP_MAX_ATTEMPTS:-120}"
delay_seconds="${RAG_BOOTSTRAP_RETRY_SECONDS:-2}"
input_path="${RAG_BOOTSTRAP_INPUT:-/opt/agentweb/data/processed/growth_records.text_only.jsonl}"

if [ ! -f "${input_path}" ]; then
  echo "[bootstrap] normalized retrieval input is missing: ${input_path}" >&2
  exit 1
fi

echo "[bootstrap] initializing PostgreSQL memory and LangGraph checkpointer"
attempt=1
while ! python -m src.cli.rag_memory_init; do
  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "[bootstrap] PostgreSQL memory did not become ready after ${attempts} attempts." >&2
    exit 1
  fi
  echo "[bootstrap] waiting for PostgreSQL memory (${attempt}/${attempts})"
  attempt=$((attempt + 1))
  sleep "${delay_seconds}"
done

echo "[bootstrap] checking MiniLM Milvus collection"
attempt=1
while :; do
  if [ "${RAG_RECREATE_MILVUS:-0}" = "1" ]; then
    python -m src.cli.rag_bootstrap --input "${input_path}" --recreate && break
  else
    python -m src.cli.rag_bootstrap --input "${input_path}" --auto-recreate && break
  fi

  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "[bootstrap] Milvus collection did not become ready after ${attempts} attempts." >&2
    exit 1
  fi
  echo "[bootstrap] waiting for Milvus (${attempt}/${attempts})"
  attempt=$((attempt + 1))
  sleep "${delay_seconds}"
done

echo "[bootstrap] initialization complete"
