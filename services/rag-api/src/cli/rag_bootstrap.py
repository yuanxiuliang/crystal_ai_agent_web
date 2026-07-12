from __future__ import annotations

import argparse
from pathlib import Path

from ..config import PROJECT_ROOT, settings
from ..retrieval.embedding import get_default_embedding_client
from ..retrieval.milvus_store import MilvusGrowthStore, iter_normalized_records
from .rag_ingest import ingest_file


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "growth_records.text_only.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-bootstrap",
        description="Ensure the MiniLM-backed Milvus collection is ready for RAG.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Normalized text JSONL input path.")
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before importing all records.",
    )
    parser.add_argument(
        "--auto-recreate",
        action="store_true",
        help="Recreate automatically when the collection schema does not match the embedding model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    expected_count = sum(1 for _ in iter_normalized_records(input_path))
    if args.limit is not None:
        expected_count = min(expected_count, args.limit)

    embedding_client = get_default_embedding_client()
    actual_embedding_dim = embedding_client.dimension
    if actual_embedding_dim != settings.embedding_dim:
        print(
            "[embedding] model output dimension does not match configuration. "
            f"model_output={actual_embedding_dim} configured={settings.embedding_dim}"
        )
        return 2

    store = MilvusGrowthStore()
    collection = store.collection_name
    if args.recreate:
        print(f"[milvus] recreating collection={collection}")
        return ingest_file(
            input_path,
            batch_size=args.batch_size,
            limit=args.limit,
            recreate=True,
        )

    if not store.client.has_collection(collection):
        print(f"[milvus] collection={collection} is missing; importing {expected_count} records")
        return ingest_file(
            input_path,
            batch_size=args.batch_size,
            limit=args.limit,
            recreate=False,
        )

    actual_count = store.count()
    dense_dim = _dense_dimension(store)
    print(
        f"[milvus] collection={collection} rows={actual_count} "
        f"dense_dim={dense_dim} expected_rows={expected_count} expected_dim={settings.embedding_dim}"
    )

    if dense_dim != settings.embedding_dim:
        print(
            "[milvus] collection schema does not match the embedding model. "
            f"collection_dim={dense_dim} model_dim={settings.embedding_dim}"
        )
        if args.auto_recreate:
            print(f"[milvus] rebuilding collection={collection} for the configured embedding model")
            return ingest_file(
                input_path,
                batch_size=args.batch_size,
                limit=args.limit,
                recreate=True,
            )
        print("[milvus] run rag-bootstrap --recreate to rebuild it.")
        return 2
    if actual_count == 0:
        print(f"[milvus] collection is empty; importing {expected_count} records")
        return ingest_file(
            input_path,
            batch_size=args.batch_size,
            limit=args.limit,
            recreate=False,
        )
    if actual_count != expected_count:
        print(
            "[milvus] row count does not match the input file. "
            f"collection_rows={actual_count} input_rows={expected_count}"
        )
        if args.auto_recreate:
            print(f"[milvus] rebuilding collection={collection} to refresh incomplete data")
            return ingest_file(
                input_path,
                batch_size=args.batch_size,
                limit=args.limit,
                recreate=True,
            )
        print("[milvus] run rag-bootstrap --recreate to refresh the collection.")
        return 2

    print("[milvus] collection is ready; skipping import")
    return 0


def _dense_dimension(store: MilvusGrowthStore) -> int | None:
    description = store.client.describe_collection(store.collection_name)
    for field in description.get("fields", []):
        if field.get("name") != settings.milvus_dense_field:
            continue
        params = field.get("params") or field.get("type_params") or {}
        try:
            return int(params["dim"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
