from __future__ import annotations

from dataclasses import replace
from typing import Any

from .method_aliases import normalize_method
from .schemas import NormalizedGrowthRecord, NormalizedReactant
from .temperature import coerce_float, format_number, format_temperature_c


REACTANT_ROLE_LABELS = {
    "raw": "starting material",
    "adtv": "additive",
    "raw_adtv": "starting material and additive",
}


def parse_doi(sample_id: str) -> str:
    return sample_id.split("::", 1)[0].strip() if sample_id else ""


def normalize_reactants(raw_reactants: object) -> tuple[NormalizedReactant, ...]:
    if not isinstance(raw_reactants, list):
        return ()
    reactants: list[NormalizedReactant] = []
    for item in raw_reactants:
        if not isinstance(item, dict):
            continue
        name = str(item.get("n") or "").strip()
        if not name:
            continue
        raw_role = str(item.get("type") or "").strip()
        role = REACTANT_ROLE_LABELS.get(raw_role, raw_role or "reactant")
        ratio = coerce_float(item.get("r"))
        reactants.append(NormalizedReactant(name=name, role=role, ratio=ratio))
    return tuple(reactants)


def normalize_growth(method_normalized: str, raw_growth: object) -> dict[str, Any]:
    growth = raw_growth if isinstance(raw_growth, dict) else {}
    normalized: dict[str, Any] = {}
    duration = coerce_float(growth.get("dur"))
    if duration is not None:
        normalized["duration_hours"] = duration

    if method_normalized == "chemical vapor transport":
        source = coerce_float(growth.get("T_src"))
        crystal = coerce_float(growth.get("T_crys"))
        if source is not None:
            normalized["source_temperature_c"] = source
        if crystal is not None:
            normalized["crystal_temperature_c"] = crystal
        if source is not None and crystal is not None:
            normalized["temperature_gradient_c"] = abs(source - crystal)
        return normalized

    start = coerce_float(growth.get("T_s"))
    end = coerce_float(growth.get("T_e"))
    if start is not None:
        normalized["start_temperature_c"] = start
    if end is not None:
        normalized["end_temperature_c"] = end
    if start is not None and end is not None:
        normalized["cooling_range_c"] = abs(start - end)
    return normalized


def normalize_record(raw: dict[str, Any]) -> NormalizedGrowthRecord:
    sample_id = str(raw.get("sample_id") or "").strip()
    formula = str(raw.get("formula") or "").strip()
    method = normalize_method(str(raw.get("method") or "").strip())
    reactants = normalize_reactants(raw.get("reactants"))
    growth = normalize_growth(method.normalized, raw.get("growth"))
    record = NormalizedGrowthRecord(
        record_id=sample_id,
        formula=formula,
        doi=parse_doi(sample_id),
        method_raw=method.raw,
        method_normalized=method.normalized,
        method_aliases=method.aliases,
        method_zh=method.zh_name,
        reactants=reactants,
        growth=growth,
        comment=str(raw.get("comment") or "").strip(),
        raw_record=raw,
    )
    return replace(record, normalized_text=build_normalized_text(record))


def build_normalized_text(record: NormalizedGrowthRecord) -> str:
    sentences: list[str] = []

    if record.doi:
        sentences.append(f"DOI: {record.doi}. ")
    material = f"{record.formula} single crystals" if record.formula else "the single crystals"
    method_text = _method_text(record)
    sentences.append(f"For {material}, this record uses {method_text} for crystal growth. ")

    reactants_text = _reactants_text(record)
    if reactants_text:
        sentences.append(reactants_text)

    growth_text = _growth_natural_text(record)
    if growth_text:
        sentences.append(growth_text)

    if record.comment:
        sentences.append(f"Additional note: {_clean_comment(record.comment)}. ")

    return "".join(sentences).strip()


def _method_text(record: NormalizedGrowthRecord) -> str:
    if record.method_normalized == "chemical vapor transport":
        return "chemical vapor transport (CVT)"
    if record.method_normalized == "flux growth":
        return "flux growth"
    return record.method_normalized or record.method_raw or "an unknown growth method"


def _reactants_text(record: NormalizedGrowthRecord) -> str:
    raw = [item for item in record.reactants if item.role == "starting material"]
    additives = [item for item in record.reactants if item.role == "additive"]
    dual = [item for item in record.reactants if item.role == "starting material and additive"]

    sentences: list[str] = []
    if raw:
        sentences.append(
            f"The starting {_plural_noun('material', raw)} "
            f"{_was_were(raw)} {_format_reactants(raw)}"
        )

    if additives:
        additive_text = _format_reactants(additives)
        if record.method_normalized == "chemical vapor transport":
            sentences.append(
                f"The transport {_plural_noun('agent', additives)} "
                f"{_was_were(additives)} {additive_text}"
            )
        else:
            sentences.append(
                f"The flux {_plural_noun('agent', additives)}, "
                f"{_plural_noun('solvent', additives)}, or {_plural_noun('additive', additives)} "
                f"{_was_were(additives)} {additive_text}"
            )

    if dual:
        if record.method_normalized == "chemical vapor transport":
            if len(dual) == 1:
                dual_role = "a starting material and a transport agent"
            else:
                dual_role = "starting materials and transport agents"
        else:
            if len(dual) == 1:
                dual_role = "a starting material and a flux agent, solvent, or additive"
            else:
                dual_role = "starting materials and flux agents, solvents, or additives"
        sentences.append(f"{_format_reactants(dual)} served as both {dual_role}")

    if not sentences:
        return ""
    return ". ".join(sentences) + ". "


def _format_reactants(reactants: list[NormalizedReactant]) -> str:
    parts = []
    for item in reactants:
        if item.ratio is None:
            parts.append(item.name)
        else:
            parts.append(f"{item.name} with ratio {format_number(item.ratio)}")
    return _join_human(parts)


def _growth_natural_text(record: NormalizedGrowthRecord) -> str:
    growth = record.growth
    parts: list[str] = []

    if record.method_normalized == "chemical vapor transport":
        source = format_temperature_c(growth.get("source_temperature_c"))
        crystal = format_temperature_c(growth.get("crystal_temperature_c"))
        if source and crystal:
            parts.append(
                f"the source-zone temperature was {source} "
                f"and the crystal-growth-zone temperature was {crystal}"
            )
        elif source:
            parts.append(f"the source-zone temperature was {source}")
        elif crystal:
            parts.append(f"the crystal-growth-zone temperature was {crystal}")
    else:
        start = format_temperature_c(growth.get("start_temperature_c"))
        end = format_temperature_c(growth.get("end_temperature_c"))
        if start and end:
            parts.append(f"the growth was carried out from {start} to {end}")
        elif start:
            parts.append(f"the growth temperature was {start}")
        elif end:
            parts.append(f"the ending or cooling temperature was {end}")

    duration = _format_duration_hours_only(growth.get("duration_hours"))
    if duration:
        parts.append(f"the growth duration was {duration}")
    if not parts:
        return ""

    return "Growth conditions: " + "; ".join(parts) + ". "


def _format_duration_hours_only(value: object) -> str | None:
    hours = coerce_float(value)
    if hours is None:
        return None
    return f"{format_number(hours)} h"


def _was_were(items: list[NormalizedReactant]) -> str:
    return "was" if len(items) == 1 else "were"


def _plural_noun(noun: str, items: list[NormalizedReactant]) -> str:
    return noun if len(items) == 1 else f"{noun}s"


def _join_human(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _clean_comment(comment: str) -> str:
    return " ".join(comment.replace("：", ":").split())
