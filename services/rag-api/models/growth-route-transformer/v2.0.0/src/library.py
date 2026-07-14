from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from pymatgen.core import Composition
from pymatgen.core.periodic_table import DummySpecies

THIS_DIR = Path(__file__).resolve().parent
V2_ROOT = THIS_DIR.parent
PROJECT_ROOT = V2_ROOT.parent.parent

DATA_RAW_DIR = V2_ROOT / "data" / "raw"
DATA_MAPPINGS_DIR = V2_ROOT / "data" / "mappings"
DATA_CLEANED_DIR = V2_ROOT / "data" / "cleaned"
DATA_ROUTES_DIR = V2_ROOT / "data" / "routes"
DATA_SPLITS_DIR = V2_ROOT / "data" / "splits"
LIB_RAW_DIR = V2_ROOT / "lib" / "rawLib"
LIB_CONFIG_DIR = V2_ROOT / "lib" / "config"
FEATURES_DIR = V2_ROOT / "features"
MODELS_DIR = V2_ROOT / "models"
REPORTS_DIR = V2_ROOT / "reports"

DEFAULT_EXTERNAL_INPUT = DATA_RAW_DIR / "merged_all.jsonl"

TEMP_MIN = -50.0
TEMP_MAX = 3000.0
TEMP_BIN_SIZE = 10.0
DUR_BIN_SIZE = 5.0

FORMULA_TOKENS = {"(", ")", "[", "]"}
METHOD_MAP = {"flux": "Flux", "cvt": "CVT"}
SPECIAL_INPUT_TOKENS = ["[PAD]", "[UNK]"]
SPECIAL_OUTPUT_TOKENS = ["[PAD]", "[UNK]"]

ROUTE_SPECIAL_TOKENS = [
    "[BOS]",
    "[EOS]",
    "[NULL]",
    "[FLUX]",
    "[CVT]",
    "[REACT]",
    "[/REACT]",
    "[TYPE]",
    "[/TYPE]",
    "[END_REACTS]",
    "<T_s>",
    "</T_s>",
    "<T_e>",
    "</T_e>",
    "<T_src>",
    "</T_src>",
    "<T_crys>",
    "</T_crys>",
    "<dur>",
    "</dur>",
    "raw",
    "adtv",
]

REACTANT_NAME_TO_FORMULA_PATH = DATA_MAPPINGS_DIR / "reactant_name_to_formula.json"
REACTANT_DROP_EXACT_PATH = DATA_MAPPINGS_DIR / "reactant_drop_exact.json"
REACTANT_DROP_SUBSTRINGS_PATH = DATA_MAPPINGS_DIR / "reactant_drop_substrings.json"
REACTANT_TRAILING_SUFFIXES_PATH = DATA_MAPPINGS_DIR / "reactant_trailing_suffixes.json"
REACTANT_STRIP_PARENS_RE = re.compile(r"\([^()]*\)")


def ensure_v2_dirs() -> None:
    for path in (
        DATA_RAW_DIR,
        DATA_MAPPINGS_DIR,
        DATA_CLEANED_DIR,
        DATA_ROUTES_DIR,
        DATA_SPLITS_DIR,
        LIB_RAW_DIR,
        LIB_CONFIG_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_reactant_name_to_formula_map() -> Dict[str, str]:
    if not REACTANT_NAME_TO_FORMULA_PATH.exists():
        raise FileNotFoundError(
            f"reactant name mapping json not found: {REACTANT_NAME_TO_FORMULA_PATH}"
        )
    payload = load_json(REACTANT_NAME_TO_FORMULA_PATH)
    return {str(k).strip().lower(): str(v).strip() for k, v in payload.items()}


@lru_cache(maxsize=1)
def get_reactant_drop_exact_set() -> set[str]:
    if not REACTANT_DROP_EXACT_PATH.exists():
        raise FileNotFoundError(
            f"reactant drop-exact json not found: {REACTANT_DROP_EXACT_PATH}"
        )
    with REACTANT_DROP_EXACT_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"reactant drop-exact json must be a list: {REACTANT_DROP_EXACT_PATH}")
    return {str(item).strip().lower() for item in payload if str(item).strip()}


@lru_cache(maxsize=1)
def get_reactant_drop_substrings() -> Tuple[str, ...]:
    if not REACTANT_DROP_SUBSTRINGS_PATH.exists():
        raise FileNotFoundError(
            f"reactant drop-substrings json not found: {REACTANT_DROP_SUBSTRINGS_PATH}"
        )
    with REACTANT_DROP_SUBSTRINGS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(
            f"reactant drop-substrings json must be a list: {REACTANT_DROP_SUBSTRINGS_PATH}"
        )
    return tuple(str(item).strip().lower() for item in payload if str(item).strip())


@lru_cache(maxsize=1)
def get_reactant_trailing_suffixes() -> Tuple[str, ...]:
    if not REACTANT_TRAILING_SUFFIXES_PATH.exists():
        raise FileNotFoundError(
            f"reactant trailing-suffixes json not found: {REACTANT_TRAILING_SUFFIXES_PATH}"
        )
    with REACTANT_TRAILING_SUFFIXES_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(
            f"reactant trailing-suffixes json must be a list: {REACTANT_TRAILING_SUFFIXES_PATH}"
        )
    return tuple(str(item).strip().lower() for item in payload if str(item).strip())


def normalize_formula_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("：", ":")
        .replace("／", "/")
        .replace("－", "-")
    )
    if "·" in text:
        parts = text.split("·")
        base = parts[0]
        suffix = []
        for seg in parts[1:]:
            match = re.fullmatch(
                r"(?P<coef>\d+(?:\.\d+)?)?(?P<solv>H2O|D2O)",
                seg,
                flags=re.IGNORECASE,
            )
            if match:
                coef = match.group("coef") or ""
                solv = match.group("solv").upper().replace("D2O", "H2O")
                suffix.append(f"({solv}){coef}" if coef else f"({solv})")
            else:
                suffix.append("·" + seg)
        text = base + "".join(suffix)
    return text


def _format_fraction(num: int, den: int) -> str:
    value = num / den
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalize_numeric_fractions(text: str) -> str:
    if "/" not in text or "%" in text:
        return text

    def _repl(match: re.Match[str]) -> str:
        num = int(match.group("num"))
        den = int(match.group("den"))
        if den == 0:
            return match.group(0)
        return _format_fraction(num, den)

    return re.sub(r"(?P<num>\d+)\/(?P<den>\d+)", _repl, text)


def normalize_formula_candidate(value: Any) -> str:
    text = normalize_formula_text(value)
    if not text:
        return ""
    return normalize_numeric_fractions(text)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_optional_float(value: Any) -> Tuple[bool, Optional[float]]:
    if value is None:
        return True, None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return True, None
        return True, float(value)
    text = str(value).strip()
    if not text or text.lower() == "null":
        return True, None
    try:
        return True, float(text)
    except Exception:
        return False, None


def parse_composition(text: str) -> Optional[Composition]:
    try:
        comp = Composition(text)
    except Exception:
        normalized = normalize_numeric_fractions(text)
        if normalized == text:
            return None
        try:
            comp = Composition(normalized)
        except Exception:
            return None
    if any(isinstance(e, DummySpecies) for e in comp.elements):
        return None
    return comp


def ordered_elements_from_formula(formula: str) -> List[str]:
    tokens = re.findall(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[\(\)\[\]]", formula)
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        if re.fullmatch(r"[A-Z][a-z]?", token):
            if token not in seen:
                ordered.append(token)
                seen.add(token)
    return ordered


def formula_elements_set(formula: str) -> List[str]:
    comp = parse_composition(formula)
    if comp is None:
        return []
    return sorted(e.symbol for e in comp.elements)


def tokenize_formula(formula: str) -> List[str]:
    tokens = re.findall(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[\(\)\[\]]", formula)
    out: List[str] = []
    for i, token in enumerate(tokens):
        if token == "1":
            prev = tokens[i - 1] if i > 0 else ""
            if re.fullmatch(r"[A-Z][a-z]?|[\)\]]", prev):
                continue
        out.append(token)
    return out


def normalize_method(value: Any) -> str:
    text = str(value).strip().lower()
    return METHOD_MAP.get(text, "")


def reactant_key(text: str) -> str:
    return normalize_formula_text(text).lower()


def is_isotope_like_reactant(text: str) -> bool:
    return bool(re.fullmatch(r"\d+[A-Z][a-z]?\d*", text))


def strip_reactant_parenthetical_annotations(text: str) -> str:
    current = text
    while True:
        updated = REACTANT_STRIP_PARENS_RE.sub("", current)
        updated = updated.strip()
        if updated == current:
            return updated
        current = updated


def strip_trailing_reactant_suffixes(text: str) -> str:
    candidate = text
    changed = True
    suffixes = get_reactant_trailing_suffixes()
    while changed and candidate:
        changed = False
        lowered = candidate.lower()
        for suffix in suffixes:
            if lowered.endswith(suffix):
                candidate = candidate[: -len(suffix)].rstrip("-/:,")
                changed = True
                break
    return candidate


def try_parse_reactant_formula(text: str) -> Optional[str]:
    candidate = normalize_formula_candidate(text)
    if not candidate:
        return None
    if parse_composition(candidate) is None:
        return None
    return candidate


def normalize_single_reactant_name(text: str) -> Optional[str]:
    candidate = normalize_formula_text(text)
    if not candidate:
        return None

    reactant_map = get_reactant_name_to_formula_map()
    drop_exact = get_reactant_drop_exact_set()
    drop_substrings = get_reactant_drop_substrings()

    direct = try_parse_reactant_formula(candidate)
    if direct is not None:
        return direct

    key = reactant_key(candidate)
    if key in reactant_map:
        mapped = reactant_map[key]
        return mapped if parse_composition(mapped) is not None else None
    if "graphite" in key:
        return "C"
    if key in drop_exact or is_isotope_like_reactant(candidate):
        return None
    if any(part in key for part in drop_substrings):
        return None
    if key.endswith("solution") and key not in reactant_map:
        return None

    without_parens = strip_reactant_parenthetical_annotations(candidate)
    if without_parens != candidate:
        direct = try_parse_reactant_formula(without_parens)
        if direct is not None:
            return direct
        key = reactant_key(without_parens)
        if key in reactant_map:
            mapped = reactant_map[key]
            return mapped if parse_composition(mapped) is not None else None
        if "graphite" in key:
            return "C"
        candidate = without_parens

    stripped = strip_trailing_reactant_suffixes(candidate)
    if stripped != candidate:
        direct = try_parse_reactant_formula(stripped)
        if direct is not None:
            return direct
        key = reactant_key(stripped)
        if key in reactant_map:
            mapped = reactant_map[key]
            return mapped if parse_composition(mapped) is not None else None
        if "graphite" in key:
            return "C"

    return None


def expand_reactant_names(text: str) -> Optional[List[str]]:
    candidate = normalize_formula_text(text)
    if not candidate:
        return None

    single = normalize_single_reactant_name(candidate)
    if single is not None:
        return [single]

    simplified = strip_reactant_parenthetical_annotations(candidate)
    simplified = strip_trailing_reactant_suffixes(simplified)
    if not simplified:
        return None

    if "/" in simplified and "%" in simplified:
        return None

    if not any(sep in simplified for sep in ("-", "/", ":")):
        return None

    raw_parts = [part.strip() for part in re.split(r"[-/:]", simplified) if part.strip()]
    if len(raw_parts) <= 1:
        return None

    cleaned_parts: List[str] = []
    for part in raw_parts:
        cleaned = normalize_single_reactant_name(part)
        if cleaned is None:
            return None
        cleaned_parts.append(cleaned)
    return cleaned_parts


def bucket_temperature(value: float) -> Optional[str]:
    if value is None:
        return None
    if not (TEMP_MIN <= value < TEMP_MAX):
        return None
    idx = int((value - TEMP_MIN) // TEMP_BIN_SIZE)
    return f"TEMP_BIN_{idx}"


def bucket_duration(value: Optional[float]) -> str:
    if value is None:
        return "[NULL]"
    idx = max(0, int(value // DUR_BIN_SIZE))
    return f"DUR_BIN_{idx}"


def normalize_growth_structure(method: str, growth: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    if not isinstance(growth, dict):
        return None
    if method == "Flux":
        ok_t1, t1 = normalize_optional_float(growth.get("T_s"))
        ok_t2, t2 = normalize_optional_float(growth.get("T_e"))
        ok_dur, dur = normalize_optional_float(growth.get("dur"))
        if not (ok_t1 and ok_t2 and ok_dur):
            return None
        return {"T_s": t1, "T_e": t2, "dur": dur}
    if method == "CVT":
        ok_t1, t1 = normalize_optional_float(growth.get("T_src"))
        ok_t2, t2 = normalize_optional_float(growth.get("T_crys"))
        ok_dur, dur = normalize_optional_float(growth.get("dur"))
        if not (ok_t1 and ok_t2 and ok_dur):
            return None
        return {"T_src": t1, "T_crys": t2, "dur": dur}
    return None


def normalize_growth(method: str, growth: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    if not isinstance(growth, dict):
        return None
    if method == "Flux":
        t1 = safe_float(growth.get("T_s"))
        t2 = safe_float(growth.get("T_e"))
        if t1 is None or t2 is None:
            return None
        if bucket_temperature(t1) is None or bucket_temperature(t2) is None:
            return None
        return {"T_s": t1, "T_e": t2, "dur": safe_float(growth.get("dur"))}
    if method == "CVT":
        t1 = safe_float(growth.get("T_src"))
        t2 = safe_float(growth.get("T_crys"))
        if t1 is None or t2 is None:
            return None
        if bucket_temperature(t1) is None or bucket_temperature(t2) is None:
            return None
        return {"T_src": t1, "T_crys": t2, "dur": safe_float(growth.get("dur"))}
    return None


def reactants_cover_target_elements(
    reactants: Sequence[Dict[str, Any]],
    target_elements: Sequence[str],
) -> bool:
    target_set = set(target_elements)
    covered: set[str] = set()
    for reactant in reactants:
        covered.update(str(el) for el in reactant.get("elements", []))
    return target_set.issubset(covered)


def _reactant_type_from_raw(value: str, reactant_formula: str, target_elements: Sequence[str]) -> str:
    low = str(value).strip().lower()
    if low == "adtv":
        return "adtv"
    if low == "raw":
        return "raw"
    if low == "raw_adtv":
        reactant_elements = set(formula_elements_set(reactant_formula))
        return "raw" if reactant_elements.intersection(target_elements) else "adtv"
    return "raw"


def dedupe_reactants_by_role(reactants: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ordered_keys: List[Tuple[str, str]] = []
    for reactant in reactants:
        name = str(reactant["name"])
        r_type = str(reactant["type"])
        key = (name, r_type)
        if key not in unique:
            unique[key] = {
                "name": name,
                "type": r_type,
                "r": reactant.get("r"),
                "elements": sorted(set(str(el) for el in reactant.get("elements", []))),
            }
            ordered_keys.append(key)
            continue
        existing = unique[key]
        if existing.get("r") is None and reactant.get("r") is not None:
            existing["r"] = reactant.get("r")
        existing["elements"] = sorted(
            set(str(el) for el in existing.get("elements", []))
            | set(str(el) for el in reactant.get("elements", []))
        )
    return [unique[key] for key in ordered_keys]


def split_reactants_by_type(
    reactants: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_reactants: List[Dict[str, Any]] = []
    adtv_reactants: List[Dict[str, Any]] = []
    for reactant in reactants:
        if reactant.get("type") == "adtv":
            adtv_reactants.append(dict(reactant))
        else:
            raw_reactants.append(dict(reactant))
    return raw_reactants, adtv_reactants


def normalize_reactants(reactants: Sequence[Dict[str, Any]], formula_std: str) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(reactants, list) or not reactants:
        return None
    target_order = ordered_elements_from_formula(formula_std)
    target_set = set(target_order)
    normalized: List[Dict[str, Any]] = []
    for reactant in reactants:
        if not isinstance(reactant, dict):
            return None
        expanded_names = expand_reactant_names(str(reactant.get("n", "")))
        if not expanded_names:
            return None
        for name in expanded_names:
            elements = formula_elements_set(name)
            r_type = _reactant_type_from_raw(reactant.get("type", "raw"), name, target_set)
            normalized.append(
                {
                    "name": name,
                    "type": r_type,
                    "r": safe_float(reactant.get("r")),
                    "elements": elements,
                }
            )
    return sort_reactants(dedupe_reactants_by_role(normalized), target_order)


def sort_reactants(reactants: Sequence[Dict[str, Any]], target_order: Sequence[str]) -> List[Dict[str, Any]]:
    index_map = {el: idx for idx, el in enumerate(target_order)}

    def raw_sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
        matched = [index_map[el] for el in item["elements"] if el in index_map]
        pos = min(matched) if matched else 10**6
        return (pos, item["name"])

    raw_reactants, adtv_reactants = split_reactants_by_type(reactants)
    return [
        *sorted(raw_reactants, key=raw_sort_key),
        *sorted(adtv_reactants, key=lambda item: item["name"]),
    ]


def build_reactant_catalog(
    records: Sequence[Dict[str, Any]],
    *,
    reactant_type: str,
) -> List[Dict[str, Any]]:
    counter: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for reactant in rec.get("reactants", []):
            if reactant.get("type") != reactant_type:
                continue
            formula = str(reactant["name"])
            item = counter.setdefault(
                formula,
                {
                    "formula": formula,
                    "count": 0,
                    "elements": sorted(str(el) for el in reactant.get("elements", [])),
                },
            )
            item["count"] += 1
    return sorted(counter.values(), key=lambda x: (-int(x["count"]), str(x["formula"])))


def serialize_route(method: str, reactants: Sequence[Dict[str, Any]], growth: Dict[str, Optional[float]]) -> List[str]:
    tokens = ["[BOS]", "[FLUX]" if method == "Flux" else "[CVT]"]
    for reactant in reactants:
        tokens.extend(
            [
                "[REACT]",
                reactant["name"],
                "[TYPE]",
                reactant["type"],
                "[/TYPE]",
                "[/REACT]",
            ]
        )
    tokens.append("[END_REACTS]")
    if method == "Flux":
        tokens.extend(
            [
                "<T_s>",
                bucket_temperature(growth["T_s"]),
                "</T_s>",
                "<T_e>",
                bucket_temperature(growth["T_e"]),
                "</T_e>",
            ]
        )
    else:
        tokens.extend(
            [
                "<T_src>",
                bucket_temperature(growth["T_src"]),
                "</T_src>",
                "<T_crys>",
                bucket_temperature(growth["T_crys"]),
                "</T_crys>",
            ]
        )
    tokens.extend(["<dur>", bucket_duration(growth.get("dur")), "</dur>", "[EOS]"])
    return tokens


def build_vocab(tokens_iter: Iterable[Sequence[str]], special_tokens: Sequence[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    for tokens in tokens_iter:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: token for token, idx in vocab.items()}


def route_token_meta(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reagent_tokens = sorted({r["name"] for rec in records for r in rec["reactants"]})
    temp_tokens = sorted(
        {
            token
            for rec in records
            for token in rec["route_tokens"]
            if token.startswith("TEMP_BIN_")
        }
    )
    dur_tokens = sorted(
        {
            token
            for rec in records
            for token in rec["route_tokens"]
            if token.startswith("DUR_BIN_")
        }
    )
    return {
        "reagent_tokens": reagent_tokens,
        "temp_tokens": temp_tokens,
        "dur_tokens": dur_tokens,
        "type_tokens": ["raw", "adtv"],
        "method_tokens": ["[FLUX]", "[CVT]"],
    }


def group_split_by_formula(
    records: Sequence[Dict[str, Any]],
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import random

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["formula_std"], []).append(rec)
    formulas = list(grouped.keys())
    rnd = random.Random(seed)
    rnd.shuffle(formulas)
    n_val = max(1, int(round(len(formulas) * val_ratio)))
    val_formulas = set(formulas[:n_val])
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for formula, group in grouped.items():
        if formula in val_formulas:
            val_rows.extend(group)
        else:
            train_rows.extend(group)
    return train_rows, val_rows


@dataclass
class BeamState:
    tokens: List[int]
    logprob: float
    stage: str
    method_token: Optional[str]
    covered: set
    reactant_count: int


def topk_display_probabilities(logprobs: Sequence[float]) -> List[float]:
    if not logprobs:
        return []
    m = max(logprobs)
    exps = [math.exp(x - m) for x in logprobs]
    s = sum(exps) or 1.0
    return [x / s for x in exps]


def allowed_next_tokens(
    state: BeamState,
    *,
    target_elements: Sequence[str],
    output_vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
    token_meta: Dict[str, Any],
    reactant_element_map: Dict[str, List[str]],
) -> List[int]:
    target_set = set(target_elements)
    max_reactants = len(target_set) + 2
    covered = set(state.covered)
    missing = target_set - covered
    reagent_ids = [output_vocab[t] for t in token_meta["reagent_tokens"] if t in output_vocab]
    temp_ids = [output_vocab[t] for t in token_meta["temp_tokens"] if t in output_vocab]
    dur_ids = [output_vocab[t] for t in token_meta["dur_tokens"] if t in output_vocab]

    def _ids(tokens: Sequence[str]) -> List[int]:
        return [output_vocab[t] for t in tokens if t in output_vocab]

    if state.stage == "method":
        return _ids(["[FLUX]", "[CVT]"])
    if state.stage == "react_open":
        if state.reactant_count >= max_reactants:
            return _ids(["[END_REACTS]"]) if not missing else []
        allowed = _ids(["[REACT]"])
        if not missing and state.reactant_count > 0:
            allowed += _ids(["[END_REACTS]"])
        return allowed
    if state.stage == "react_name":
        allowed: List[int] = []
        for idx in reagent_ids:
            token = inverse_vocab[idx]
            elements = set(reactant_element_map.get(token, []))
            if not missing or elements.intersection(missing):
                allowed.append(idx)
        return allowed
    if state.stage == "type_open":
        return _ids(["[TYPE]"])
    if state.stage == "type_value":
        return _ids(["raw", "adtv"])
    if state.stage == "type_close":
        return _ids(["[/TYPE]"])
    if state.stage == "react_close":
        return _ids(["[/REACT]"])
    if state.stage == "temp1_open":
        return _ids(["<T_s>"] if state.method_token == "[FLUX]" else ["<T_src>"])
    if state.stage == "temp1_value":
        return temp_ids
    if state.stage == "temp1_close":
        return _ids(["</T_s>"] if state.method_token == "[FLUX]" else ["</T_src>"])
    if state.stage == "temp2_open":
        return _ids(["<T_e>"] if state.method_token == "[FLUX]" else ["<T_crys>"])
    if state.stage == "temp2_value":
        return temp_ids
    if state.stage == "temp2_close":
        return _ids(["</T_e>"] if state.method_token == "[FLUX]" else ["</T_crys>"])
    if state.stage == "dur_open":
        return _ids(["<dur>"])
    if state.stage == "dur_value":
        return dur_ids + _ids(["[NULL]"])
    if state.stage == "dur_close":
        return _ids(["</dur>"])
    if state.stage == "eos":
        return _ids(["[EOS]"])
    return []


def next_stage(current_stage: str, emitted_token: str, method_token: Optional[str]) -> str:
    if current_stage == "method":
        return "react_open"
    if current_stage == "react_open":
        return "react_name" if emitted_token == "[REACT]" else "temp1_open"
    if current_stage == "react_name":
        return "type_open"
    if current_stage == "type_open":
        return "type_value"
    if current_stage == "type_value":
        return "type_close"
    if current_stage == "type_close":
        return "react_close"
    if current_stage == "react_close":
        return "react_open"
    if current_stage == "temp1_open":
        return "temp1_value"
    if current_stage == "temp1_value":
        return "temp1_close"
    if current_stage == "temp1_close":
        return "temp2_open"
    if current_stage == "temp2_open":
        return "temp2_value"
    if current_stage == "temp2_value":
        return "temp2_close"
    if current_stage == "temp2_close":
        return "dur_open"
    if current_stage == "dur_open":
        return "dur_value"
    if current_stage == "dur_value":
        return "dur_close"
    if current_stage == "dur_close":
        return "eos"
    if current_stage == "eos":
        return "done"
    return "done"


def advance_state(
    state: BeamState,
    emitted_token: str,
    *,
    token_id: int,
    reactant_element_map: Dict[str, List[str]],
) -> BeamState:
    method_token = state.method_token
    if state.stage == "method":
        method_token = emitted_token

    covered = set(state.covered)
    reactant_count = state.reactant_count
    if state.stage == "react_name":
        covered.update(reactant_element_map.get(emitted_token, []))
        reactant_count += 1

    return BeamState(
        tokens=state.tokens + [token_id],
        logprob=state.logprob,
        stage=next_stage(state.stage, emitted_token, method_token),
        method_token=method_token,
        covered=covered,
        reactant_count=reactant_count,
    )


def constrained_beam_search(
    model,
    *,
    formula_ids: torch.Tensor,
    target_elements: Sequence[str],
    output_vocab: Dict[str, int],
    token_meta: Dict[str, Any],
    reactant_element_map: Dict[str, List[str]],
    device: str,
    beam_size: int = 10,
    num_return_sequences: int = 3,
    max_steps: int = 128,
) -> List[Dict[str, Any]]:
    inverse_vocab = invert_vocab(output_vocab)
    bos_id = output_vocab["[BOS]"]
    src_pad_mask = formula_ids.eq(0)
    with torch.no_grad():
        memory = model.encode(formula_ids.to(device), src_pad_mask=src_pad_mask.to(device))

    beams = [
        BeamState(
            tokens=[bos_id],
            logprob=0.0,
            stage="method",
            method_token=None,
            covered=set(),
            reactant_count=0,
        )
    ]
    finished: List[BeamState] = []

    for _ in range(max_steps):
        all_candidates: List[BeamState] = []
        for state in beams:
            if state.stage == "done":
                finished.append(state)
                continue
            allowed = allowed_next_tokens(
                state,
                target_elements=target_elements,
                output_vocab=output_vocab,
                inverse_vocab=inverse_vocab,
                token_meta=token_meta,
                reactant_element_map=reactant_element_map,
            )
            if not allowed:
                continue
            tgt = torch.tensor([state.tokens], dtype=torch.long, device=device)
            tgt_pad_mask = torch.zeros_like(tgt, dtype=torch.bool)
            with torch.no_grad():
                logits = model.decode(
                    tgt,
                    memory,
                    tgt_pad_mask=tgt_pad_mask,
                    memory_pad_mask=src_pad_mask.to(device),
                )[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            allowed_scores = [(tok_id, float(log_probs[tok_id])) for tok_id in allowed]
            allowed_scores.sort(key=lambda x: x[1], reverse=True)
            for tok_id, lp in allowed_scores[:beam_size]:
                token = inverse_vocab[tok_id]
                new_state = advance_state(
                    state,
                    token,
                    token_id=tok_id,
                    reactant_element_map=reactant_element_map,
                )
                all_candidates.append(
                    BeamState(
                        tokens=new_state.tokens,
                        logprob=state.logprob + lp,
                        stage=new_state.stage,
                        method_token=new_state.method_token,
                        covered=new_state.covered,
                        reactant_count=new_state.reactant_count,
                    )
                )

        if not all_candidates:
            break
        all_candidates.sort(key=lambda x: x.logprob, reverse=True)
        beams = all_candidates[:beam_size]
        if all(state.stage == "done" for state in beams):
            finished.extend(beams)
            break

    if not finished:
        finished = beams
    finished.sort(key=lambda x: x.logprob, reverse=True)
    top = finished[:num_return_sequences]
    display_probs = topk_display_probabilities([x.logprob for x in top])
    return [
        {
            "tokens": [inverse_vocab[idx] for idx in state.tokens],
            "logprob": state.logprob,
            "display_probability": display_probs[i],
        }
        for i, state in enumerate(top)
    ]
