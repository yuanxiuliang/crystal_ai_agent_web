from __future__ import annotations

import json
from pathlib import Path
import re
import hashlib
from typing import Any, Iterable

from pymilvus import DataType, Function, FunctionType, MilvusClient

from ..config import Settings, settings


class MilvusGrowthStore:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.client = MilvusClient(
            uri=config.milvus_uri,
            token=config.milvus_token or None,
            db_name=config.milvus_db_name,
        )

    @property
    def collection_name(self) -> str:
        return self.config.milvus_collection

    def recreate_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.create_collection()

    def create_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("record_id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("formula", DataType.VARCHAR, max_length=128)
        schema.add_field("doi", DataType.VARCHAR, max_length=256)
        schema.add_field("growth_method_normalized", DataType.VARCHAR, max_length=128)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=8192)
        schema.add_field(
            self.config.milvus_text_field,
            DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
        )
        schema.add_field(
            self.config.milvus_dense_field,
            DataType.FLOAT_VECTOR,
            dim=self.config.embedding_dim,
        )
        schema.add_field(self.config.milvus_sparse_field, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[self.config.milvus_text_field],
                output_field_names=[self.config.milvus_sparse_field],
            )
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level=self.config.milvus_consistency_level,
        )
        self.create_indexes()

    def create_indexes(self) -> None:
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            self.config.milvus_dense_field,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            self.config.milvus_sparse_field,
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        self.client.create_index(self.collection_name, index_params)
        self.client.load_collection(self.collection_name)

    def insert_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        result = self.client.insert(self.collection_name, rows)
        return int(result.get("insert_count", len(rows)))

    def flush_and_load(self) -> None:
        self.client.flush(self.collection_name)
        self.client.load_collection(self.collection_name)

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            count = int(stats.get("row_count", 0))
            if count:
                return count
        except Exception:
            pass
        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if result and "count(*)" in result[0]:
            return int(result[0]["count(*)"])
        return 0


def iter_normalized_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def to_milvus_row(
    record: dict[str, Any],
    dense_vector: list[float],
    *,
    sequence_id: int | None = None,
    config: Settings = settings,
) -> dict[str, Any]:
    text = _record_text(record)
    metadata = record.get("metadata") or {}
    doi = str(record.get("doi") or metadata.get("doi") or _extract_doi(text) or "")
    formula = str(record.get("formula") or metadata.get("formula") or _extract_formula(text) or "")
    method = str(
        record.get("method_normalized")
        or metadata.get("growth_method_normalized")
        or _extract_growth_method(text)
        or ""
    )
    record_id = str(record.get("record_id") or metadata.get("record_id") or "")
    if not record_id:
        record_id = _text_record_id(text, sequence_id)
    if not metadata:
        metadata = {
            "record_id": record_id,
            "formula": formula,
            "doi": doi,
            "growth_method_normalized": method,
        }
    return {
        "record_id": record_id,
        "formula": formula,
        "doi": doi,
        "growth_method_normalized": method,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        config.milvus_text_field: text,
        config.milvus_dense_field: dense_vector,
    }


def _record_text(record: dict[str, Any]) -> str:
    text = str(record.get("text") or record.get("normalized_text") or "").strip()
    if not text:
        raise ValueError("Input record must contain either 'text' or 'normalized_text'.")
    return text


def _text_record_id(text: str, sequence_id: int | None) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    if sequence_id is None:
        return f"text::{digest}"
    return f"text::{sequence_id:06d}::{digest}"


def _extract_doi(text: str) -> str | None:
    match = re.search(r"\bDOI:\s*([^\s.]+(?:\.[^\s.]+)*)", text)
    return match.group(1).rstrip(".") if match else None


def _extract_formula(text: str) -> str | None:
    match = re.search(r"\bFor\s+(.+?)\s+single crystals\b", text)
    return match.group(1).strip() if match else None


def _extract_growth_method(text: str) -> str | None:
    text_l = text.lower()
    if "chemical vapor transport" in text_l or "(cvt)" in text_l:
        return "chemical vapor transport"
    if "flux growth" in text_l:
        return "flux growth"
    return None
