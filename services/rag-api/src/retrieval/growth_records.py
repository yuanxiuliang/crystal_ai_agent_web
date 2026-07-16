from __future__ import annotations

import re
from typing import Any

from ..agent.state import RetrievedRecord


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


def normalize_growth_record(raw: dict[str, Any], source_file: str) -> RetrievedRecord:
    sample_id = str(raw.get("sample_id") or "")
    doi = _extract_doi(raw, sample_id)
    formula = _none_if_empty(raw.get("formula"))
    method = _none_if_empty(raw.get("method"))
    reactants = raw.get("reactants") if isinstance(raw.get("reactants"), list) else []
    growth = raw.get("growth") if isinstance(raw.get("growth"), dict) else {}

    raw_reactants = _reactant_names(reactants, {"raw"})
    additives = _reactant_names(reactants, {"adtv"})
    raw_additives = _reactant_names(reactants, {"raw_adtv"})
    all_reactants = [*raw_reactants, *additives, *raw_additives]
    temperature_program = build_temperature_program(method, growth)
    source_text = build_structured_source_text(
        record_id=sample_id,
        doi=doi,
        formula=formula,
        method=method,
        raw_reactants=raw_reactants,
        additives=additives,
        raw_additives=raw_additives,
        temperature_program=temperature_program,
        comment=_none_if_empty(raw.get("comment")),
    )

    return {
        "record_id": sample_id,
        "score": 0.0,
        "dense_score": None,
        "sparse_score": None,
        "material_formula": formula,
        "material_name": None,
        "growth_method": method,
        "temperature_program": temperature_program,
        "atmosphere": None,
        "precursors": all_reactants,
        "doi": doi,
        "source_text": source_text,
        "source_file": source_file,
        "matched_fields": [],
    }


def build_temperature_program(method: str | None, growth: dict[str, Any]) -> str | None:
    if not method:
        return None

    duration = _format_duration(growth.get("dur"))
    if method == "Flux":
        start = _format_temp(growth.get("T_s"))
        end = _format_temp(growth.get("T_e"))
        parts = ["Flux growth"]
        if start:
            parts.append(f"start/high temperature {start}")
        if end:
            parts.append(f"end/cooling temperature {end}")
        if duration:
            parts.append(f"duration {duration}")
        return "; ".join(parts) + "." if len(parts) > 1 else None

    if method == "CVT":
        src = _format_temp(growth.get("T_src"))
        crystal = _format_temp(growth.get("T_crys"))
        parts = ["CVT growth"]
        if src:
            parts.append(f"source zone {src}")
        if crystal:
            parts.append(f"crystal zone {crystal}")
        if duration:
            parts.append(f"duration {duration}")
        return "; ".join(parts) + "." if len(parts) > 1 else None

    return None


def build_structured_source_text(
    *,
    record_id: str,
    doi: str | None,
    formula: str | None,
    method: str | None,
    raw_reactants: list[str],
    additives: list[str],
    raw_additives: list[str],
    temperature_program: str | None,
    comment: str | None,
) -> str:
    parts = [f"Structured growth record {record_id}."]
    if doi:
        parts.append(f"DOI: {doi}.")
    if formula:
        parts.append(f"Material formula: {formula}.")
    if method:
        parts.append(f"Growth method: {method}.")
    if raw_reactants:
        parts.append(f"Raw reactants: {', '.join(raw_reactants)}.")
    if additives:
        parts.append(f"Additives or transport/flux agents: {', '.join(additives)}.")
    if raw_additives:
        parts.append(f"Reactants also used as additives/flux: {', '.join(raw_additives)}.")
    if temperature_program:
        parts.append(f"Temperature program: {temperature_program}")
    if comment and comment != "human_reviewed":
        parts.append(f"Comment: {comment}.")
    return " ".join(parts)


def _reactant_names(reactants: list[Any], allowed_types: set[str]) -> list[str]:
    names: list[str] = []
    for item in reactants:
        if not isinstance(item, dict):
            continue
        if item.get("type") in allowed_types and item.get("n"):
            ratio = item.get("r")
            name = str(item["n"])
            if ratio is not None:
                name = f"{name} (ratio {ratio})"
            names.append(name)
    return names


def _format_temp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{value:g} C"
    return str(value)


def _format_duration(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{value:g} h"
    return str(value)


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_doi(raw: dict[str, Any], sample_id: str) -> str | None:
    explicit_doi = _none_if_empty(raw.get("doi")) or _none_if_empty(raw.get("DOI"))
    if explicit_doi:
        return explicit_doi

    prefix = sample_id.split("::", 1)[0] if "::" in sample_id else ""
    return prefix if _DOI_PATTERN.fullmatch(prefix) else None
