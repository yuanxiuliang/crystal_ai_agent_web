from __future__ import annotations

from ...prediction import (
    PredictionExecutionRequest,
    PredictionInputError,
    PredictionService,
    PredictionUnavailableError,
    PredictionValidationError,
)
from ..state import GrowthRAGState
from ..utils import error, trace


async def run_prediction(state: GrowthRAGState, prediction: PredictionService) -> dict:
    eligibility = state["prediction_eligibility"]
    formula = eligibility["formula"] if eligibility else None
    if not eligibility or not eligibility["eligible"] or not formula:
        return {
            "prediction_error": "Prediction fallback was attempted without eligible input.",
            "errors": [
                error(
                    "run_prediction",
                    "ineligible_prediction",
                    "Prediction input is ineligible.",
                    True,
                )
            ],
        }
    try:
        result = await prediction.predict(
            PredictionExecutionRequest(
                user_id=state["user_id"],
                session_id=state["session_id"],
                message_id=state["message_id"],
                formula=formula,
                source="retrieval_fallback",
            )
        )
    except (PredictionInputError, PredictionUnavailableError, PredictionValidationError) as exc:
        return {
            "prediction_error": str(exc),
            "errors": [error("run_prediction", type(exc).__name__, str(exc), True)],
            "trace": [trace("run_prediction", "failed", {"error": type(exc).__name__})],
        }
    return {
        "prediction_result": result.as_dict(),
        "selected_evidence_kind": "model_prediction",
        "citations": [],
        "trace": [
            trace(
                "run_prediction",
                "predicted",
                {
                    "prediction_run_id": result.prediction_run_id,
                    "model_id": result.model.model_id,
                    "model_version": result.model.model_version,
                    "route_count": len(result.routes),
                },
            )
        ],
    }
