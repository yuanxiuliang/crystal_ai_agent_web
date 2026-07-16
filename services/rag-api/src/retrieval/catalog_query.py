from __future__ import annotations

import re
from typing import Literal, TypedDict


AggregateQueryKind = Literal[
    "element_method_distribution",
    "method_material_catalog",
    "transport_agent_material_catalog",
    "reactant_product_catalog",
]


class AggregateReactantFilter(TypedDict):
    name: str
    roles: list[str]


class AggregateQuery(TypedDict):
    kind: AggregateQueryKind
    label: str
    element: str | None
    growth_method: str | None
    reactants: list[AggregateReactantFilter]


# The catalog deals with chemical formula strings, so this must be a chemical-token parser,
# not a broad word regex. It intentionally supports the decimal compositions in the corpus.
_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_FORMULA_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?)+(?![A-Za-z])"
)
_ELEMENT_FAMILY_RE = re.compile(
    r"(?:含\s*)?([A-Z][a-z]?)\s*(?:基化合物|基材料|基体系|元素体系|体系|基)"
)

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_TRANSPORT_TERMS = ("传输剂", "运输剂", "气相输运剂", "transport agent")
_RAW_TERMS = ("起始原料", "原料", "反应物", "前驱体")
_MATERIAL_CATALOG_TERMS = ("哪些化合物", "哪些材料", "什么化合物", "什么材料", "适用于", "适合", "覆盖")


def normalize_chemical_token(value: str) -> str:
    return "".join(value.translate(_SUBSCRIPT_TRANSLATION).split())


def formula_elements(value: str) -> list[str]:
    """Return unique, ordered element symbols for a formula-like corpus field."""
    normalized = normalize_chemical_token(value)
    symbols = re.findall(r"[A-Z][a-z]?", normalized)
    residue = re.sub(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[().+\-·]", "", normalized)
    if residue or not symbols or any(symbol not in _ELEMENTS for symbol in symbols):
        return []
    return list(dict.fromkeys(symbols))


def extract_chemical_tokens(value: str) -> list[str]:
    normalized = value.translate(_SUBSCRIPT_TRANSLATION)
    tokens: list[str] = []
    for match in _FORMULA_TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if token in {"CVT", "DOI"} or not formula_elements(token):
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens))


def normalize_reactant_name(value: str) -> str:
    return normalize_chemical_token(value)


def detect_aggregate_query(user_message: str) -> AggregateQuery | None:
    """Recognize deterministic corpus questions that do not need one target formula.

    The LLM may still improve ordinary material questions, but it must not decide whether a
    structured reverse lookup is allowed. These query kinds have explicit, auditable filters.
    """
    lowered = user_message.lower()
    has_transport = any(term in user_message or term in lowered for term in _TRANSPORT_TERMS)
    if has_transport:
        agent = _transport_agent(user_message)
        if agent:
            return {
                "kind": "transport_agent_material_catalog",
                "label": f"{agent} 作为 CVT 传输剂",
                "element": None,
                "growth_method": "chemical vapor transport",
                "reactants": [
                    {"name": agent, "roles": ["additive", "raw_and_additive"]}
                ],
            }

    if any(term in user_message for term in _RAW_TERMS):
        reactants = extract_chemical_tokens(user_message)
        if reactants:
            return {
                "kind": "reactant_product_catalog",
                "label": "、".join(reactants) + " 作为起始原料",
                "element": None,
                "growth_method": _method_hint(user_message),
                "reactants": [
                    {"name": value, "roles": ["raw", "raw_and_additive"]}
                    for value in reactants
                ],
            }

    family_match = _ELEMENT_FAMILY_RE.search(user_message)
    if family_match:
        element = family_match.group(1)
        if element in _ELEMENTS:
            return {
                "kind": "element_method_distribution",
                "label": f"含 {element} 的目标化学式",
                "element": element,
                "growth_method": _method_hint(user_message),
                "reactants": [],
            }

    method = _method_hint(user_message)
    if method and any(term in user_message or term in lowered for term in _MATERIAL_CATALOG_TERMS):
        method_label = "Flux" if method == "flux growth" else "CVT"
        return {
            "kind": "method_material_catalog",
            "label": f"{method_label} 方法的已报道材料",
            "element": None,
            "growth_method": method,
            "reactants": [],
        }
    return None


def _transport_agent(user_message: str) -> str | None:
    if "碘" in user_message:
        # In the source corpus, the Chinese phrase "碘传输剂" denotes molecular iodine.
        return "I2"
    for token in extract_chemical_tokens(user_message):
        if token != "CVT":
            return normalize_reactant_name(token)
    return None


def _method_hint(value: str) -> str | None:
    lowered = value.lower()
    if "flux" in lowered or "助熔剂" in value or "助溶剂" in value:
        return "flux growth"
    if "cvt" in lowered or "气相输运" in value or "气相传输" in value:
        return "chemical vapor transport"
    return None
