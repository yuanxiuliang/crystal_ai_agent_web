from __future__ import annotations

from typing import Any

from .contracts import PredictionModelInfo, PredictionRoute, PredictionValidationError


def _range(value: Any, *, field: str, unit_key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PredictionValidationError(f"Prediction route field {field} is missing.")
    bounds = value.get(unit_key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise PredictionValidationError(f"Prediction route field {field} has an invalid range.")
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError(
            f"Prediction route field {field} has a non-numeric range."
        ) from exc
    if low >= high:
        raise PredictionValidationError(
            f"Prediction route field {field} must have increasing bounds."
        )
    return {"token": str(value.get("token") or ""), unit_key: [bounds[0], bounds[1]]}


def _validated_growth(method: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PredictionValidationError("Prediction route is missing growth conditions.")
    result: dict[str, Any] = {}
    if method == "Flux":
        result["T_s"] = _range(value.get("T_s"), field="T_s", unit_key="range_c")
        result["T_e"] = _range(value.get("T_e"), field="T_e", unit_key="range_c")
    else:
        result["T_src"] = _range(value.get("T_src"), field="T_src", unit_key="range_c")
        result["T_crys"] = _range(value.get("T_crys"), field="T_crys", unit_key="range_c")
    duration = value.get("dur")
    result["dur"] = None if duration is None else _range(duration, field="dur", unit_key="range_h")
    return result


def _reactants(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PredictionValidationError(f"Prediction route {field} must be a list.")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            raise PredictionValidationError(
                f"Prediction route {field} contains an invalid reactant."
            )
        elements = item.get("elements")
        if not isinstance(elements, list):
            raise PredictionValidationError(
                f"Prediction route {field} reactant is missing elements."
            )
        result.append(
            {
                "name": str(item["name"]),
                "type": str(item.get("type") or ""),
                "r": item.get("r"),
                "elements": [str(element) for element in elements],
            }
        )
    return result


def validate_prediction_routes(
    raw_routes: Any,
    *,
    target_elements: list[str],
    model: PredictionModelInfo,
    requested_count: int,
) -> list[PredictionRoute]:
    if not isinstance(raw_routes, list) or not raw_routes:
        raise PredictionValidationError("Prediction model returned no route candidates.")
    if len(raw_routes) > requested_count:
        raise PredictionValidationError("Prediction model returned more routes than requested.")

    expected_elements = set(target_elements)
    validated: list[PredictionRoute] = []
    for index, route in enumerate(raw_routes, start=1):
        if not isinstance(route, dict):
            raise PredictionValidationError("Prediction model returned a non-object route.")
        method = str(route.get("method") or "")
        if method not in model.supported_methods or method not in {"Flux", "CVT"}:
            raise PredictionValidationError(f"Prediction route uses unsupported method: {method!r}")
        raw_reactants = _reactants(route.get("raw_reactants"), field="raw_reactants")
        additives = _reactants(route.get("adtv_reactants"), field="adtv_reactants")
        covered = {element for item in [*raw_reactants, *additives] for element in item["elements"]}
        coverage_ok = bool(route.get("element_coverage_ok")) and expected_elements.issubset(covered)
        if not coverage_ok:
            raise PredictionValidationError(
                "Prediction route fails target-element coverage validation."
            )
        try:
            rank = int(route.get("rank"))
            weight = float(route.get("display_probability"))
        except (TypeError, ValueError) as exc:
            raise PredictionValidationError(
                "Prediction route rank or relative ranking is invalid."
            ) from exc
        if rank != index or weight < 0:
            raise PredictionValidationError("Prediction route ordering is invalid.")
        validated.append(
            PredictionRoute(
                rank=rank,
                relative_rank_weight=weight,
                method=method,  # type: ignore[arg-type]
                raw_reactants=raw_reactants,
                additives=additives,
                growth=_validated_growth(method, route.get("growth")),
                element_coverage_ok=True,
            )
        )
    return validated
