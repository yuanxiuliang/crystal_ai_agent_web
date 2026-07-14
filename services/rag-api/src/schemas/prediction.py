from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=160)
    message_id: str | None = Field(default=None, max_length=160)
    formula: str = Field(min_length=1, max_length=160)


class PredictionModelResponse(BaseModel):
    model_id: str
    model_version: str
    artifact_digest: str
    supported_methods: list[str]
    parameter_count: int


class PredictionRouteResponse(BaseModel):
    rank: int
    relative_rank_weight: float
    method: str
    raw_reactants: list[dict]
    additives: list[dict]
    growth: dict
    element_coverage_ok: bool


class PredictionResponse(BaseModel):
    prediction_run_id: str
    source: str
    formula: str
    formula_std: str
    formula_tokens: list[str]
    target_elements: list[str]
    unknown_formula_tokens: list[str]
    routes: list[PredictionRouteResponse]
    model: PredictionModelResponse
    warnings: list[str]
    runtime_ms: int
