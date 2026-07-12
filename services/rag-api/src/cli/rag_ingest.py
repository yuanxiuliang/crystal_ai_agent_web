from __future__ import annotations

import argparse
from pathlib import Path

from ..config import PROJECT_ROOT, settings
from ..retrieval.embedding import get_default_embedding_client
from ..retrieval.milvus_store import MilvusGrowthStore, iter_normalized_records, to_milvus_row


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "growth_records.text_only.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest normalized growth records into Milvus with dense and BM25 indexes.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Normalized JSONL input path.")
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and recreate collection first."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return ingest_file(
        Path(args.input),
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
    )


def ingest_file(
    input_path: Path,
    *,
    batch_size: int = settings.embedding_batch_size,
    limit: int | None = None,
    recreate: bool = False,
) -> int:
    embedding_client = get_default_embedding_client()
    actual_dim = embedding_client.dimension
    if actual_dim != settings.embedding_dim:
        raise RuntimeError(
            "Embedding dimension mismatch before Milvus write: "
            f"model_output={actual_dim}, configured={settings.embedding_dim}."
        )

    store = MilvusGrowthStore()
    if recreate:
        store.recreate_collection()
    else:
        store.create_collection()

    batch: list[dict] = []
    total = 0
    last_reported = 0

    for sequence_id, record in enumerate(iter_normalized_records(input_path), start=1):
        if limit is not None and total + len(batch) >= limit:
            break
        batch.append({"sequence_id": sequence_id, "record": record})
        if len(batch) >= batch_size:
            total += ingest_batch(store, embedding_client, batch)
            if total - last_reported >= 200:
                print(f"inserted={total}")
                last_reported = total
            batch = []
    if batch:
        total += ingest_batch(store, embedding_client, batch)
    if total != last_reported:
        print(f"inserted={total}")

    store.flush_and_load()
    print(f"collection={settings.milvus_collection}")
    print(f"row_count={store.count()}")
    return 0


def ingest_batch(store: MilvusGrowthStore, embedding_client, records: list[dict]) -> int:
    texts = [_record_text(item["record"]) for item in records]
    vectors = embedding_client.embed_texts(texts)
    rows = [
        to_milvus_row(item["record"], vector, sequence_id=item["sequence_id"])
        for item, vector in zip(records, vectors, strict=True)
    ]
    return store.insert_rows(rows)


def _record_text(record: dict) -> str:
    text = str(record.get("text") or record.get("normalized_text") or "").strip()
    if not text:
        raise ValueError("Input record must contain either 'text' or 'normalized_text'.")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
