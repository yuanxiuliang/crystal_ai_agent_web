from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


PredictionSource = Literal["explicit_prediction", "retrieval_fallback"]


class PredictionError(RuntimeError):
    """Base error for a prediction request that can be rendered without a traceback."""


class PredictionInputError(PredictionError):
    """The user did not provide a formula accepted by the deployed model."""


class PredictionUnavailableError(PredictionError):
    """The local model runtime or its verified artifacts are unavailable."""


class PredictionValidationError(PredictionError):
    """The model emitted a result that violates the deployment contract."""


@dataclass(frozen=True)
class FormulaValidation:
    formula: str
    formula_std: str
    formula_tokens: list[str]
    target_elements: list[str]
    unknown_formula_tokens: list[str]


@dataclass(frozen=True)
class PredictionExecutionRequest:
    user_id: str
    formula: str
    session_id: str | None = None
    message_id: str | None = None
    source: PredictionSource = "explicit_prediction"


@dataclass(frozen=True)
class PredictionModelInfo:
    model_id: str
    model_version: str
    artifact_digest: str
    supported_methods: list[str]
    parameter_count: int


@dataclass(frozen=True)
class PredictionRoute:
    rank: int
    relative_rank_weight: float
    method: Literal["Flux", "CVT"]
    raw_reactants: list[dict[str, Any]]
    additives: list[dict[str, Any]]
    growth: dict[str, Any]
    element_coverage_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    prediction_run_id: str
    source: PredictionSource
    formula: str
    formula_std: str
    formula_tokens: list[str]
    target_elements: list[str]
    unknown_formula_tokens: list[str]
    routes: list[PredictionRoute]
    model: PredictionModelInfo
    warnings: list[str]
    runtime_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_run_id": self.prediction_run_id,
            "source": self.source,
            "formula": self.formula,
            "formula_std": self.formula_std,
            "formula_tokens": self.formula_tokens,
            "target_elements": self.target_elements,
            "unknown_formula_tokens": self.unknown_formula_tokens,
            "routes": [route.as_dict() for route in self.routes],
            "model": asdict(self.model),
            "warnings": self.warnings,
            "runtime_ms": self.runtime_ms,
        }
