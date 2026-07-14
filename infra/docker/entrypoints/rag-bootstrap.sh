#!/bin/sh
set -eu

attempts="${RAG_BOOTSTRAP_MAX_ATTEMPTS:-120}"
delay_seconds="${RAG_BOOTSTRAP_RETRY_SECONDS:-2}"
input_path="${RAG_BOOTSTRAP_INPUT:-/opt/agentweb/data/processed/growth_records.text_only.jsonl}"
raw_input_path="${GROWTH_RECORDS_PATH:-/opt/agentweb/rawData.jsonl}"

if [ ! -f "${input_path}" ]; then
  if [ ! -f "${raw_input_path}" ]; then
    echo "[bootstrap] raw retrieval input is missing: ${raw_input_path}" >&2
    exit 1
  fi
  echo "[bootstrap] generating normalized retrieval input from versioned raw data"
  python -m src.cli.rag_normalize \
    --input "${raw_input_path}" \
    --output "${input_path}" \
    --output-format text-only \
    --preview 0
fi

if [ ! -f "${input_path}" ]; then
  echo "[bootstrap] normalized retrieval input was not generated: ${input_path}" >&2
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
