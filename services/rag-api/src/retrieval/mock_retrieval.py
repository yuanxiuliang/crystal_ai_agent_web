from __future__ import annotations

from ..agent.state import RetrievalPlan, RetrievedRecord


MOCK_RECORDS: list[RetrievedRecord] = [
    {
        "record_id": "mock-mn3gan-001",
        "score": 0.91,
        "dense_score": 0.88,
        "sparse_score": 0.94,
        "material_formula": "Mn3GaN",
        "material_name": "manganese gallium nitride",
        "growth_method": "flux growth",
        "temperature_program": "heated to 1100 C and slowly cooled to 900 C",
        "atmosphere": "argon",
        "precursors": ["Mn", "Ga"],
        "doi": "mock-doi-001",
        "source_text": "Mn3GaN single crystals were grown by a flux method under argon. The mixture was heated to 1100 C and slowly cooled to 900 C.",
        "source_file": "mock_growth_records.jsonl",
        "matched_fields": ["material_formula", "growth_method", "temperature_program", "atmosphere"],
    },
    {
        "record_id": "mock-bi2se3-001",
        "score": 0.84,
        "dense_score": 0.82,
        "sparse_score": 0.86,
        "material_formula": "Bi2Se3",
        "material_name": "bismuth selenide",
        "growth_method": "Bridgman method",
        "temperature_program": "melted above 850 C and cooled along a temperature gradient",
        "atmosphere": "vacuum",
        "precursors": ["Bi", "Se"],
        "doi": "mock-doi-002",
        "source_text": "Bi2Se3 crystals were prepared by the Bridgman method in evacuated quartz ampoules.",
        "source_file": "mock_growth_records.jsonl",
        "matched_fields": ["material_formula", "growth_method"],
    },
]


class MockRetrievalService:
    async def retrieve(self, plan: RetrievalPlan) -> list[RetrievedRecord]:
        formula = plan["filters"].get("material_formula")
        query = plan["query_text"].lower()
        records = MOCK_RECORDS
        if formula:
            records = [record for record in records if record["material_formula"] == formula]
        if not records:
            records = [
                record
                for record in MOCK_RECORDS
                if record["material_formula"] and record["material_formula"].lower() in query
            ]
        return records[: plan["top_k"]]

