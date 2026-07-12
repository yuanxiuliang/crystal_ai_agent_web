from __future__ import annotations

import json
import re
from typing import Any, Literal

from pymilvus import AnnSearchRequest, RRFRanker

from ..agent.state import RetrievalFilters, RetrievalPlan, RetrievedRecord
from ..config import Settings, settings
from ..ingestion.method_aliases import normalize_method
from .embedding import EmbeddingClient, get_default_embedding_client
from .milvus_store import MilvusGrowthStore
from .query_expansion import expand_query_for_bm25
from .rrf import RankedHit, reciprocal_rank_fusion


SearchMode = Literal["bm25", "dense", "hybrid"]


class MilvusHybridRetriever:
    def __init__(
        self,
        config: Settings = settings,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        self.store = MilvusGrowthStore(config)
        self.embedding_client = embedding_client or get_default_embedding_client(config)

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        top_k: int = 5,
        trace: bool = False,
        filters: RetrievalFilters | None = None,
        relax_filters_if_empty: bool = True,
    ) -> list[RetrievedRecord]:
        filter_expr = self._build_filter_expr(filters)
        if mode == "bm25":
            hits = self.search_bm25(query, self.config.milvus_bm25_top_k, filter_expr=filter_expr)
            records = [
                self._to_retrieved_record(hit.entity, hit.score, None, hit.score)
                for hit in hits[:top_k]
            ]
            if records or not filter_expr or not relax_filters_if_empty:
                return records
            return self.search(
                query,
                mode=mode,
                top_k=top_k,
                trace=trace,
                filters=None,
                relax_filters_if_empty=False,
            )
        if mode == "dense":
            hits = self.search_dense(query, self.config.milvus_dense_top_k, filter_expr=filter_expr)
            records = [
                self._to_retrieved_record(hit.entity, hit.score, hit.score, None)
                for hit in hits[:top_k]
            ]
            if records or not filter_expr or not relax_filters_if_empty:
                return records
            return self.search(
                query,
                mode=mode,
                top_k=top_k,
                trace=trace,
                filters=None,
                relax_filters_if_empty=False,
            )
        records = self.search_hybrid(query, top_k=top_k, trace=trace, filter_expr=filter_expr)
        if records or not filter_expr or not relax_filters_if_empty:
            return records
        return self.search(
            query,
            mode=mode,
            top_k=top_k,
            trace=trace,
            filters=None,
            relax_filters_if_empty=False,
        )

    def search_bm25(
        self,
        query: str,
        limit: int,
        *,
        filter_expr: str | None = None,
    ) -> list[RankedHit]:
        bm25_query = expand_query_for_bm25(query)
        results = self.store.client.search(
            collection_name=self.config.milvus_collection,
            data=[bm25_query],
            filter=filter_expr or "",
            anns_field=self.config.milvus_sparse_field,
            search_params={"metric_type": "BM25", "params": {}},
            limit=limit,
            output_fields=self._output_fields(),
        )
        return self._ranked_hits(results[0], "bm25")

    def search_dense(
        self,
        query: str,
        limit: int,
        *,
        filter_expr: str | None = None,
    ) -> list[RankedHit]:
        vector = self.embedding_client.embed_query(expand_query_for_bm25(query))
        results = self.store.client.search(
            collection_name=self.config.milvus_collection,
            data=[vector],
            filter=filter_expr or "",
            anns_field=self.config.milvus_dense_field,
            search_params={"metric_type": "COSINE", "params": {}},
            limit=limit,
            output_fields=self._output_fields(),
        )
        return self._ranked_hits(results[0], "dense")

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        trace: bool = False,
        filter_expr: str | None = None,
    ) -> list[RetrievedRecord]:
        bm25_query = expand_query_for_bm25(query)
        dense_vector = self.embedding_client.embed_query(bm25_query)
        requests = [
            AnnSearchRequest(
                data=[bm25_query],
                anns_field=self.config.milvus_sparse_field,
                param={"metric_type": "BM25", "params": {}},
                limit=self.config.milvus_bm25_top_k,
                filter=filter_expr,
            ),
            AnnSearchRequest(
                data=[dense_vector],
                anns_field=self.config.milvus_dense_field,
                param={"metric_type": "COSINE", "params": {}},
                limit=self.config.milvus_dense_top_k,
                filter=filter_expr,
            ),
        ]
        try:
            results = self.store.client.hybrid_search(
                collection_name=self.config.milvus_collection,
                reqs=requests,
                ranker=RRFRanker(k=self.config.milvus_rrf_k),
                limit=top_k,
                output_fields=self._output_fields(),
            )
            records = [
                self._to_retrieved_record(
                    hit.get("entity", {}),
                    hit.get("distance", 0.0),
                    None,
                    None,
                )
                for hit in results[0]
            ]
            if trace:
                for record in records:
                    record["matched_fields"].append(f"bm25_query={bm25_query}")
                    if filter_expr:
                        record["matched_fields"].append(f"filter={filter_expr}")
            return records
        except Exception:
            bm25_hits = self.search_bm25(
                query,
                self.config.milvus_bm25_top_k,
                filter_expr=filter_expr,
            )
            dense_hits = self.search_dense(
                query,
                self.config.milvus_dense_top_k,
                filter_expr=filter_expr,
            )
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_hits, "dense": dense_hits},
                rrf_k=self.config.milvus_rrf_k,
            )
            records = []
            for hit in fused[:top_k]:
                dense_score = hit.debug.get("dense_score")
                bm25_score = hit.debug.get("bm25_score")
                record = self._to_retrieved_record(hit.entity, hit.score, dense_score, bm25_score)
                if trace:
                    record["matched_fields"].extend(
                        [
                            f"bm25_rank={hit.debug.get('bm25_rank')}",
                            f"dense_rank={hit.debug.get('dense_rank')}",
                            f"bm25_query={bm25_query}",
                        ]
                    )
                    if filter_expr:
                        record["matched_fields"].append(f"filter={filter_expr}")
                records.append(record)
            return records

    def _build_filter_expr(self, filters: RetrievalFilters | None) -> str | None:
        if not filters:
            return None
        clauses: list[str] = []
        formula = self._clean_filter_value(filters.get("material_formula"))
        if formula:
            clauses.append(f'formula == "{self._escape_filter_string(formula)}"')
        doi = self._clean_filter_value(filters.get("doi"))
        if doi:
            clauses.append(f'doi == "{self._escape_filter_string(doi)}"')
        method = self._normalize_growth_method_filter(filters.get("growth_method"))
        if method:
            clauses.append(
                f'growth_method_normalized == "{self._escape_filter_string(method)}"'
            )
        return " and ".join(clauses) or None

    def _normalize_growth_method_filter(self, method: str | None) -> str | None:
        value = self._clean_filter_value(method)
        if not value:
            return None
        value_l = value.lower()
        if "化学气相输运" in value or "气相输运" in value or value_l == "cvt":
            return "chemical vapor transport"
        if "助熔剂" in value or value_l == "flux":
            return "flux growth"
        return normalize_method(value).normalized

    def _clean_filter_value(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _escape_filter_string(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _ranked_hits(self, hits: list[dict[str, Any]], source: str) -> list[RankedHit]:
        ranked = []
        for index, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            ranked.append(
                RankedHit(
                    record_id=str(entity.get("record_id") or hit.get("id")),
                    score=float(hit.get("distance", 0.0)),
                    rank=index,
                    source=source,
                    entity=entity,
                )
            )
        return ranked

    def _to_retrieved_record(
        self,
        entity: dict[str, Any],
        score: float,
        dense_score: float | None,
        sparse_score: float | None,
    ) -> RetrievedRecord:
        metadata = self._metadata(entity)
        text = str(entity.get(self.config.milvus_text_field) or "")
        return {
            "record_id": str(entity.get("record_id") or metadata.get("record_id") or ""),
            "score": float(score),
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "material_formula": str(entity.get("formula") or metadata.get("formula") or "") or None,
            "material_name": None,
            "growth_method": str(
                entity.get("growth_method_normalized")
                or metadata.get("growth_method_normalized")
                or ""
            )
            or None,
            "temperature_program": self._temperature_program(metadata, text),
            "atmosphere": None,
            "precursors": [],
            "doi": str(entity.get("doi") or metadata.get("doi") or "") or None,
            "source_text": text,
            "source_file": "milvus",
            "matched_fields": [],
        }

    def _metadata(self, entity: dict[str, Any]) -> dict[str, Any]:
        raw = entity.get("metadata_json") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _temperature_program(self, metadata: dict[str, Any], text: str) -> str | None:
        keys = [
            "source_temperature_c",
            "crystal_temperature_c",
            "start_temperature_c",
            "end_temperature_c",
            "duration_hours",
        ]
        parts = [f"{key}={metadata[key]}" for key in keys if key in metadata]
        if parts:
            return "; ".join(parts)
        match = re.search(r"Growth conditions:\s*(.+?)(?:\.\s|$)", text)
        if match:
            return match.group(1).strip()
        return None

    def _output_fields(self) -> list[str]:
        return [
            "record_id",
            "formula",
            "doi",
            "growth_method_normalized",
            "metadata_json",
            self.config.milvus_text_field,
        ]


class MilvusHybridRetrievalService:
    def __init__(self, config: Settings = settings) -> None:
        self.retriever = MilvusHybridRetriever(config)

    async def retrieve(self, plan: RetrievalPlan) -> list[RetrievedRecord]:
        mode = plan.get("retrieval_mode", "hybrid")
        if mode == "sparse":
            search_mode: SearchMode = "bm25"
        elif mode == "dense":
            search_mode = "dense"
        else:
            search_mode = "hybrid"
        return self.retriever.search(
            plan["query_text"],
            mode=search_mode,
            top_k=plan["top_k"],
            trace=True,
            filters=plan.get("filters"),
            relax_filters_if_empty=plan.get("relax_filters_if_empty", True),
        )
