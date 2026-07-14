from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from library import (
    FEATURES_DIR,
    LIB_CONFIG_DIR,
    LIB_RAW_DIR,
    MODELS_DIR,
    constrained_beam_search,
    load_json,
    normalize_formula_candidate,
    ordered_elements_from_formula,
    parse_composition,
    reactants_cover_target_elements,
    topk_display_probabilities,
    tokenize_formula,
)
from network import DEFAULT_GENERATION_CANDIDATES, ModelConfig, RouteTransformer


@dataclass
class PredictionContext:
    model: RouteTransformer
    input_vocab: Dict[str, int]
    output_vocab: Dict[str, int]
    token_meta: Dict[str, Any]
    bucket_cfg: Dict[str, Any]
    reactant_element_map: Dict[str, List[str]]
    input_unk: int
    device: str
    checkpoint_path: str


def _format_number(value: float) -> int | float:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return int(rounded)
    return float(value)


def _decode_temp_bin(token: str, bucket_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not token.startswith("TEMP_BIN_"):
        raise ValueError(f"not a temperature bin token: {token}")
    idx = int(token.split("_")[-1])
    low = float(bucket_cfg["temp_min"]) + idx * float(bucket_cfg["temp_bin_size"])
    high = low + float(bucket_cfg["temp_bin_size"])
    return {
        "token": token,
        "range_c": [_format_number(low), _format_number(high)],
    }


def _decode_duration_bin(token: str, bucket_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if token == "[NULL]":
        return None
    if not token.startswith("DUR_BIN_"):
        raise ValueError(f"not a duration bin token: {token}")
    idx = int(token.split("_")[-1])
    low = idx * float(bucket_cfg["dur_bin_size"])
    high = low + float(bucket_cfg["dur_bin_size"])
    return {
        "token": token,
        "range_h": [_format_number(low), _format_number(high)],
    }


def _format_range_text(bounds: List[int | float], unit: str) -> str:
    if len(bounds) != 2:
        raise ValueError(f"range must have length 2, got: {bounds}")
    return f"{bounds[0]}-{bounds[1]} {unit}"


def _range_midpoint(bounds: List[int | float]) -> float:
    if len(bounds) != 2:
        raise ValueError(f"range must have length 2, got: {bounds}")
    return float(bounds[0] + bounds[1]) / 2.0


def route_growth_to_scalar_growth(method: str, growth: Dict[str, Any]) -> Dict[str, Optional[float]]:
    if method == "Flux":
        return {
            "T_s": _range_midpoint(growth["T_s"]["range_c"]),
            "T_e": _range_midpoint(growth["T_e"]["range_c"]),
            "dur": None if growth["dur"] is None else _range_midpoint(growth["dur"]["range_h"]),
        }
    return {
        "T_src": _range_midpoint(growth["T_src"]["range_c"]),
        "T_crys": _range_midpoint(growth["T_crys"]["range_c"]),
        "dur": None if growth["dur"] is None else _range_midpoint(growth["dur"]["range_h"]),
    }


def route_identity_key(route: Dict[str, Any]) -> tuple[str, frozenset[str], frozenset[str]]:
    raw_names = frozenset(str(x["name"]) for x in route["raw_reactants"])
    adtv_names = frozenset(str(x["name"]) for x in route["adtv_reactants"])
    return (str(route["method"]), raw_names, adtv_names)


def route_completeness_score(route: Dict[str, Any]) -> tuple[int, float]:
    scalar_growth = route_growth_to_scalar_growth(str(route["method"]), route["growth"])
    non_null_fields = sum(1 for value in scalar_growth.values() if value is not None)
    return (non_null_fields, float(route["logprob"]))


def dedupe_and_select_routes(
    routes: List[Dict[str, Any]],
    *,
    final_count: int,
) -> List[Dict[str, Any]]:
    best_by_identity: Dict[tuple[str, frozenset[str], frozenset[str]], Dict[str, Any]] = {}
    for route in routes:
        key = route_identity_key(route)
        incumbent = best_by_identity.get(key)
        if incumbent is None or route_completeness_score(route) > route_completeness_score(incumbent):
            best_by_identity[key] = route

    selected = sorted(
        best_by_identity.values(),
        key=lambda x: float(x["logprob"]),
        reverse=True,
    )[:final_count]
    display_probs = topk_display_probabilities([float(x["logprob"]) for x in selected])
    for rank, (route, prob) in enumerate(zip(selected, display_probs), start=1):
        route["rank"] = rank
        route["display_probability"] = float(prob)
    return selected


def _route_to_jsonl_record(
    *,
    formula: str,
    formula_std: str,
    route: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "formula": formula,
        "formula_std": formula_std,
        "rank": route["rank"],
        "display_probability": route["display_probability"],
        "method": route["method"],
        "raw_reactants": route["raw_reactants"],
        "adtv_reactants": route["adtv_reactants"],
        "growth": route_growth_to_scalar_growth(route["method"], route["growth"]),
    }


def _route_to_paragraph(route: Dict[str, Any]) -> str:
    raw_text = "、".join(x["name"] for x in route["raw_reactants"]) if route["raw_reactants"] else "无"
    adtv_text = "、".join(x["name"] for x in route["adtv_reactants"]) if route["adtv_reactants"] else "无"
    prob_text = f"{float(route['display_probability']):.4f}"

    if route["method"] == "Flux":
        t_start = route_growth_to_scalar_growth(route["method"], route["growth"])["T_s"]
        t_end = route_growth_to_scalar_growth(route["method"], route["growth"])["T_e"]
        temp_text = f"T_s 约为 {t_start:.1f} C，T_e 约为 {t_end:.1f} C"
    else:
        t_src = route_growth_to_scalar_growth(route["method"], route["growth"])["T_src"]
        t_crys = route_growth_to_scalar_growth(route["method"], route["growth"])["T_crys"]
        temp_text = f"T_src 约为 {t_src:.1f} C，T_crys 约为 {t_crys:.1f} C"

    scalar_growth = route_growth_to_scalar_growth(route["method"], route["growth"])
    if scalar_growth["dur"] is None:
        dur_text = "时长未给出明确预测"
    else:
        dur_text = f"时长约为 {scalar_growth['dur']:.1f} h"
    return (
        f"候选路线 {route['rank']}：模型推荐采用 {route['method']} 方法，"
        f"raw 原料为 {raw_text}，添加剂为 {adtv_text}；"
        f"温度条件预测为 {temp_text}，{dur_text}。"
        f"该路线在候选列表中的归一化显示概率为 {prob_text}。"
    )


def _parse_route_tokens(
    tokens: List[str],
    *,
    reactant_element_map: Dict[str, List[str]],
    bucket_cfg: Dict[str, Any],
    target_elements: List[str],
) -> Dict[str, Any]:
    if not tokens or tokens[0] != "[BOS]":
        raise ValueError("route must start with [BOS]")
    if len(tokens) < 2:
        raise ValueError("route is too short")

    idx = 1
    method_token = tokens[idx]
    if method_token not in {"[FLUX]", "[CVT]"}:
        raise ValueError(f"invalid method token: {method_token}")
    method = "Flux" if method_token == "[FLUX]" else "CVT"
    idx += 1

    reactants: List[Dict[str, Any]] = []
    while idx < len(tokens):
        token = tokens[idx]
        if token == "[END_REACTS]":
            idx += 1
            break
        if token != "[REACT]":
            raise ValueError(f"expected [REACT], got {token}")
        if idx + 5 >= len(tokens):
            raise ValueError("truncated reactant block")
        name = tokens[idx + 1]
        if tokens[idx + 2] != "[TYPE]":
            raise ValueError(f"expected [TYPE], got {tokens[idx + 2]}")
        r_type = tokens[idx + 3]
        if r_type not in {"raw", "adtv"}:
            raise ValueError(f"invalid reactant type token: {r_type}")
        if tokens[idx + 4] != "[/TYPE]":
            raise ValueError(f"expected [/TYPE], got {tokens[idx + 4]}")
        if tokens[idx + 5] != "[/REACT]":
            raise ValueError(f"expected [/REACT], got {tokens[idx + 5]}")
        reactants.append(
            {
                "name": name,
                "type": r_type,
                "elements": list(reactant_element_map.get(name, [])),
            }
        )
        idx += 6
    else:
        raise ValueError("missing [END_REACTS]")

    def _expect(expected: str) -> None:
        nonlocal idx
        if idx >= len(tokens) or tokens[idx] != expected:
            actual = tokens[idx] if idx < len(tokens) else "<EOF>"
            raise ValueError(f"expected {expected}, got {actual}")
        idx += 1

    growth: Dict[str, Any] = {}
    if method == "Flux":
        _expect("<T_s>")
        growth["T_s"] = _decode_temp_bin(tokens[idx], bucket_cfg)
        idx += 1
        _expect("</T_s>")
        _expect("<T_e>")
        growth["T_e"] = _decode_temp_bin(tokens[idx], bucket_cfg)
        idx += 1
        _expect("</T_e>")
    else:
        _expect("<T_src>")
        growth["T_src"] = _decode_temp_bin(tokens[idx], bucket_cfg)
        idx += 1
        _expect("</T_src>")
        _expect("<T_crys>")
        growth["T_crys"] = _decode_temp_bin(tokens[idx], bucket_cfg)
        idx += 1
        _expect("</T_crys>")

    _expect("<dur>")
    growth["dur"] = _decode_duration_bin(tokens[idx], bucket_cfg)
    idx += 1
    _expect("</dur>")
    _expect("[EOS]")

    if idx != len(tokens):
        raise ValueError(f"unexpected trailing tokens: {tokens[idx:]}")

    raw_reactants = [
        {"name": x["name"], "type": x["type"], "r": None, "elements": x["elements"]}
        for x in reactants
        if x["type"] == "raw"
    ]
    adtv_reactants = [
        {"name": x["name"], "type": x["type"], "r": None, "elements": x["elements"]}
        for x in reactants
        if x["type"] == "adtv"
    ]
    coverage_ok = reactants_cover_target_elements(reactants, target_elements)

    return {
        "method": method,
        "raw_reactants": raw_reactants,
        "adtv_reactants": adtv_reactants,
        "growth": growth,
        "element_coverage_ok": coverage_ok,
    }


def _resolve_checkpoint(run_name: str, checkpoint: str) -> Path:
    if checkpoint:
        path = Path(checkpoint)
    else:
        path = MODELS_DIR / f"{run_name}.best.pth"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_prediction_context(
    *,
    run_name: str,
    checkpoint: str,
    device: str,
) -> PredictionContext:
    input_vocab = load_json(FEATURES_DIR / "input_vocab.json")
    output_vocab = load_json(FEATURES_DIR / "output_vocab.json")
    token_meta = load_json(LIB_CONFIG_DIR / "token_meta.json")
    bucket_cfg = load_json(LIB_CONFIG_DIR / "bucket_config.json")
    reactant_element_map = load_json(LIB_RAW_DIR / "reactant_element_map.json")

    checkpoint_path = _resolve_checkpoint(run_name, checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = RouteTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return PredictionContext(
        model=model,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        token_meta=token_meta,
        bucket_cfg=bucket_cfg,
        reactant_element_map=reactant_element_map,
        input_unk=int(input_vocab["[UNK]"]),
        device=device,
        checkpoint_path=str(checkpoint_path),
    )


def generate_routes_with_context(
    *,
    formula: str,
    beam_size: int,
    num_return_sequences: int,
    context: PredictionContext,
) -> Dict[str, Any]:
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be positive")
    raw_candidate_count = max(DEFAULT_GENERATION_CANDIDATES, num_return_sequences)
    effective_beam_size = max(int(beam_size), raw_candidate_count)

    formula_std = normalize_formula_candidate(formula)
    if not formula_std or parse_composition(formula_std) is None:
        raise ValueError(f"formula is not parseable after normalization: {formula}")
    formula_tokens = tokenize_formula(formula_std)
    target_elements = ordered_elements_from_formula(formula_std)
    unknown_formula_tokens = [tok for tok in formula_tokens if tok not in context.input_vocab]
    formula_ids = torch.tensor(
        [[int(context.input_vocab.get(tok, context.input_unk)) for tok in formula_tokens]],
        dtype=torch.long,
    )

    predictions = constrained_beam_search(
        context.model,
        formula_ids=formula_ids,
        target_elements=target_elements,
        output_vocab=context.output_vocab,
        token_meta=context.token_meta,
        reactant_element_map=context.reactant_element_map,
        device=context.device,
        beam_size=effective_beam_size,
        num_return_sequences=raw_candidate_count,
    )

    raw_routes: List[Dict[str, Any]] = []
    for pred in predictions:
        parsed = _parse_route_tokens(
            list(pred["tokens"]),
            reactant_element_map=context.reactant_element_map,
            bucket_cfg=context.bucket_cfg,
            target_elements=target_elements,
        )
        raw_routes.append(
            {
                "display_probability": float(pred["display_probability"]),
                "logprob": float(pred["logprob"]),
                **parsed,
            }
        )
    routes = dedupe_and_select_routes(raw_routes, final_count=num_return_sequences)
    return {
        "formula": formula,
        "formula_std": formula_std,
        "formula_tokens": formula_tokens,
        "unknown_formula_tokens": unknown_formula_tokens,
        "target_elements": target_elements,
        "routes": routes,
        "beam_size": effective_beam_size,
        "raw_candidate_count": raw_candidate_count,
        "checkpoint_path": context.checkpoint_path,
    }


def generate_routes(
    *,
    formula: str,
    run_name: str,
    checkpoint: str,
    beam_size: int,
    num_return_sequences: int,
    device: str,
) -> Dict[str, Any]:
    context = load_prediction_context(
        run_name=run_name,
        checkpoint=checkpoint,
        device=device,
    )
    return generate_routes_with_context(
        formula=formula,
        beam_size=beam_size,
        num_return_sequences=num_return_sequences,
        context=context,
    )


def predict(
    *,
    formula: str,
    run_name: str,
    checkpoint: str,
    beam_size: int,
    num_return_sequences: int,
    device: str,
) -> Dict[str, Any]:
    generated = generate_routes(
        formula=formula,
        run_name=run_name,
        checkpoint=checkpoint,
        beam_size=beam_size,
        num_return_sequences=num_return_sequences,
        device=device,
    )
    routes = generated["routes"]
    route_jsonl_records = [
        _route_to_jsonl_record(
            formula=generated["formula"],
            formula_std=generated["formula_std"],
            route=route,
        )
        for route in routes
    ]
    route_jsonl_lines = [
        json.dumps(record, ensure_ascii=False) for record in route_jsonl_records
    ]
    route_paragraphs = [_route_to_paragraph(route) for route in routes]

    return {
        "formula": generated["formula"],
        "formula_std": generated["formula_std"],
        "formula_tokens": generated["formula_tokens"],
        "unknown_formula_tokens": generated["unknown_formula_tokens"],
        "target_elements": generated["target_elements"],
        "routes": routes,
        "beam_size": generated["beam_size"],
        "raw_candidate_count": generated["raw_candidate_count"],
        "checkpoint_path": generated["checkpoint_path"],
        "routes_jsonl_lines": route_jsonl_lines,
        "routes_jsonl_text": "\n".join(route_jsonl_lines),
        "route_paragraphs": route_paragraphs,
        "route_paragraphs_text": "\n\n".join(route_paragraphs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 minimal Transformer prediction")
    parser.add_argument("--formula", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="v2.0.0")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_GENERATION_CANDIDATES)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--output-text", type=str, default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    result = predict(
        formula=str(args.formula),
        run_name=str(args.run_name),
        checkpoint=str(args.checkpoint),
        beam_size=int(args.beam_size),
        num_return_sequences=int(args.num_return_sequences),
        device=device,
    )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["routes_jsonl_text"] + "\n", encoding="utf-8")
    if args.output_text:
        output_path = Path(args.output_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["route_paragraphs_text"] + "\n", encoding="utf-8")
    print(result["routes_jsonl_text"])
    print()
    print(result["route_paragraphs_text"])


if __name__ == "__main__":
    main()
