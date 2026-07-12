from __future__ import annotations

import argparse
import json
from typing import Any

from ..config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-config",
        description="Inspect RAG service configuration for local workflow validation.",
    )
    parser.add_argument("--json", action="store_true", help="Print configuration as JSON.")
    parser.add_argument("--check-milvus", action="store_true", help="Try connecting to Milvus.")
    return parser


def redacted(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def config_snapshot() -> dict[str, Any]:
    return {
        "retrieval": {
            "provider": settings.retrieval_provider,
            "growth_records_path": settings.growth_records_path,
        },
        "milvus": {
            "uri": settings.milvus_uri,
            "token_set": bool(settings.milvus_token),
            "db_name": settings.milvus_db_name,
            "collection": settings.milvus_collection,
            "text_field": settings.milvus_text_field,
            "sparse_field": settings.milvus_sparse_field,
            "dense_field": settings.milvus_dense_field,
            "bm25_top_k": settings.milvus_bm25_top_k,
            "dense_top_k": settings.milvus_dense_top_k,
            "rrf_k": settings.milvus_rrf_k,
            "consistency_level": settings.milvus_consistency_level,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "base_url": settings.embedding_base_url,
            "api_key": redacted(settings.embedding_api_key),
            "model": settings.embedding_model,
            "backend": settings.embedding_backend,
            "onnx_model_path": settings.embedding_onnx_model_path,
            "tokenizer_path": settings.embedding_tokenizer_path,
            "dim": settings.embedding_dim,
            "batch_size": settings.embedding_batch_size,
            "max_length": settings.embedding_max_length,
            "timeout_seconds": settings.embedding_timeout_seconds,
            "device": settings.embedding_device,
        },
        "llm": {
            "base_url": settings.llm_base_url,
            "api_key": redacted(settings.llm_api_key),
            "model": settings.llm_model,
            "provider": settings.llm_provider,
        },
    }


def print_text(snapshot: dict[str, Any]) -> None:
    print("RAG configuration")
    for section, values in snapshot.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            print(f"{key}={value}")


def check_milvus() -> bool:
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_db_name,
        )
        collections = client.list_collections()
        print(f"\n[milvus] connected uri={settings.milvus_uri}")
        print(f"[milvus] collections={collections}")
        return True
    except Exception as exc:  # noqa: BLE001 - config CLI should report dependency/connection issues.
        print(f"\n[milvus] connection failed: {exc}")
        return False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    snapshot = config_snapshot()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print_text(snapshot)
    if args.check_milvus:
        return 0 if check_milvus() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
