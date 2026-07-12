from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent.state import RetrievalPlan, RetrievedRecord
from .growth_records import normalize_growth_record


class LocalJsonlRetrievalService:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.records = self._load_records()

    async def retrieve(self, plan: RetrievalPlan) -> list[RetrievedRecord]:
        query = plan["query_text"]
        query_l = query.lower()
        wanted_formula = plan["filters"].get("material_formula")
        wanted_method = self._normalize_method(plan["filters"].get("growth_method"), query_l)
        require_formula = bool(wanted_formula and "material_formula" in plan.get("must_have", []))
        scored: list[tuple[float, RetrievedRecord]] = []

        for record in self.records:
            if require_formula and record["material_formula"] != wanted_formula:
                continue
            score, fields = self._score_record(record, query_l, wanted_formula, wanted_method)
            if score <= 0:
                continue
            candidate = dict(record)
            candidate["score"] = score
            candidate["sparse_score"] = score
            candidate["matched_fields"] = fields
            scored.append((score, candidate))  # type: ignore[arg-type]

        scored.sort(key=lambda item: (item[0], self._completeness(item[1])), reverse=True)
        return [record for _, record in scored[: plan["top_k"]]]

    def _load_records(self) -> list[RetrievedRecord]:
        if not self.path.exists():
            return []
        records: list[RetrievedRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                records.append(normalize_growth_record(raw, self.path.name))
        return records

    def _score_record(
        self,
        record: RetrievedRecord,
        query_l: str,
        wanted_formula: str | None,
        wanted_method: str | None,
    ) -> tuple[float, list[str]]:
        score = 0.0
        fields: list[str] = []

        formula = record["material_formula"]
        if wanted_formula and formula:
            if formula == wanted_formula:
                score += 120
                fields.append("material_formula")
            elif formula.lower() == wanted_formula.lower():
                score += 100
                fields.append("material_formula")
        elif formula and formula.lower() in query_l:
            score += 90
            fields.append("material_formula")

        method = record["growth_method"]
        if wanted_method and method == wanted_method:
            score += 45
            fields.append("growth_method")
        elif method and self._method_in_query(method, query_l):
            score += 35
            fields.append("growth_method")

        if record["doi"] and record["doi"].lower() in query_l:
            score += 60
            fields.append("doi")

        for reactant in record["precursors"]:
            name = reactant.split(" (ratio", 1)[0].lower()
            if name and name in query_l:
                score += 16
                fields.append("reactants")

        source_text = record["source_text"].lower()
        for token in self._query_tokens(query_l):
            if token in source_text:
                score += 2
                fields.append("source_text")

        return score, sorted(set(fields))

    def _normalize_method(self, method: str | None, query_l: str) -> str | None:
        text = (method or "").lower()
        if text == "flux" or "flux" in query_l or "助熔剂" in query_l:
            return "Flux"
        if text == "cvt" or "cvt" in query_l or "化学气相输运" in query_l or "气相输运" in query_l:
            return "CVT"
        return method if method in {"Flux", "CVT"} else None

    def _method_in_query(self, method: str, query_l: str) -> bool:
        if method == "Flux":
            return "flux" in query_l or "助熔剂" in query_l
        if method == "CVT":
            return "cvt" in query_l or "气相输运" in query_l or "化学气相输运" in query_l
        return False

    def _query_tokens(self, query_l: str) -> list[str]:
        tokens = []
        for raw in query_l.replace("？", " ").replace("?", " ").replace(",", " ").split():
            token = raw.strip()
            if len(token) >= 3:
                tokens.append(token)
        return tokens[:12]

    def _completeness(self, record: RetrievedRecord) -> int:
        return sum(
            1
            for field in ["temperature_program", "growth_method", "doi", "source_text"]
            if record.get(field)
        )
