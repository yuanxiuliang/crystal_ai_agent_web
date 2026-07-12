from __future__ import annotations

from ..state import Citation, EvidencePack, EvidenceRecord, GrowthRAGState
from ..utils import trace


def _record_facts(record: dict) -> list[str]:
    facts: list[str] = []
    if record.get("growth_method"):
        facts.append(f"growth method: {record['growth_method']}")
    if record.get("temperature_program"):
        facts.append(f"temperature program: {record['temperature_program']}")
    if record.get("atmosphere"):
        facts.append(f"atmosphere: {record['atmosphere']}")
    if record.get("precursors"):
        facts.append(f"precursors: {', '.join(record['precursors'])}")
    return facts


async def build_evidence_pack(state: GrowthRAGState) -> dict:
    records: list[EvidenceRecord] = []
    citations: list[Citation] = []
    missing_fields: set[str] = set()
    for record in state["retrieved_records"]:
        facts = _record_facts(record)
        if not record.get("temperature_program"):
            missing_fields.add("temperature_program")
        evidence_record: EvidenceRecord = {
            "record_id": record["record_id"],
            "score": record["score"],
            "title": None,
            "material_formula": record["material_formula"],
            "growth_method": record["growth_method"],
            "key_facts": facts,
            "source_text": record["source_text"][:700],
            "doi": record["doi"],
        }
        records.append(evidence_record)
        citations.append(
            {
                "record_id": record["record_id"],
                "doi": record["doi"],
                "source_text": record["source_text"][:500],
                "score": record["score"],
                "fields_used": record["matched_fields"],
            }
        )

    pack: EvidencePack = {
        "records": records,
        "summary": f"Prepared {len(records)} evidence records.",
        "conflicts": [],
        "missing_fields": sorted(missing_fields),
    }
    return {
        "evidence_pack": pack,
        "citations": citations,
        "trace": [trace("build_evidence_pack", "built", {"count": len(records)})],
    }

