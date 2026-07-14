from __future__ import annotations

import asyncio
import time

from ..config import Settings, settings
from .contracts import (
    PredictionExecutionRequest,
    PredictionInputError,
    PredictionResult,
    PredictionUnavailableError,
    PredictionValidationError,
)
from .repository import PredictionRunRepository
from .runtime import PredictionRuntime
from .validation import validate_prediction_routes


class PredictionService:
    """The only application-facing entrypoint for formula-conditioned route prediction."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        runtime: PredictionRuntime | None = None,
        repository: PredictionRunRepository | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or PredictionRuntime(config)
        self.repository = repository or PredictionRunRepository(
            config.prediction_database_url,
            postgres_connect_timeout_seconds=config.memory_postgres_connect_timeout_seconds,
        )
        self._semaphore = asyncio.Semaphore(max(1, min(4, config.prediction_max_concurrency)))

    async def predict(self, request: PredictionExecutionRequest) -> PredictionResult:
        if not self.config.prediction_enabled:
            raise PredictionUnavailableError("Prediction is disabled by PREDICTION_ENABLED.")
        user_id = request.user_id.strip()
        if not user_id:
            raise PredictionInputError("user_id is required for prediction-run ownership.")
        if request.source not in {"explicit_prediction", "retrieval_fallback"}:
            raise PredictionInputError("Prediction source is invalid.")

        normalized_request = PredictionExecutionRequest(
            user_id=user_id,
            formula=request.formula.strip(),
            session_id=request.session_id,
            message_id=request.message_id,
            source=request.source,
        )
        formula = await asyncio.to_thread(self.runtime.validate_formula, normalized_request.formula)
        model = self.runtime.bundle.model_info
        await asyncio.to_thread(self.repository.register_model, model, self.runtime.bundle.manifest)
        run_id = await asyncio.to_thread(
            self.repository.create_run,
            normalized_request,
            formula_std=formula.formula_std,
            model=model,
        )

        started = time.perf_counter()
        try:
            async with self._semaphore:
                raw_result = await asyncio.to_thread(self.runtime.generate, formula.formula)
            routes = validate_prediction_routes(
                raw_result.get("routes"),
                target_elements=[
                    str(item) for item in raw_result.get("target_elements", formula.target_elements)
                ],
                model=model,
                requested_count=max(1, min(3, self.config.prediction_return_sequences)),
            )
            unknown = [str(item) for item in raw_result.get("unknown_formula_tokens", [])]
            warnings = self._warnings(unknown, route_count=len(routes))
            runtime_ms = int((time.perf_counter() - started) * 1000)
            result = PredictionResult(
                prediction_run_id=run_id,
                source=normalized_request.source,
                formula=str(raw_result.get("formula") or formula.formula),
                formula_std=str(raw_result.get("formula_std") or formula.formula_std),
                formula_tokens=[
                    str(item) for item in raw_result.get("formula_tokens", formula.formula_tokens)
                ],
                target_elements=[
                    str(item) for item in raw_result.get("target_elements", formula.target_elements)
                ],
                unknown_formula_tokens=unknown,
                routes=routes,
                model=model,
                warnings=warnings,
                runtime_ms=runtime_ms,
            )
            await asyncio.to_thread(
                self.repository.complete_run,
                run_id,
                result=result.as_dict(),
                warnings=warnings,
                runtime_ms=runtime_ms,
            )
            return result
        except (PredictionInputError, PredictionUnavailableError, PredictionValidationError) as exc:
            await asyncio.to_thread(
                self.repository.fail_run,
                run_id,
                error_code=type(exc).__name__,
                warnings=[str(exc)],
            )
            raise
        except Exception as exc:  # noqa: BLE001 - do not expose vendored model tracebacks as API failures.
            await asyncio.to_thread(
                self.repository.fail_run,
                run_id,
                error_code=type(exc).__name__,
                warnings=["Prediction runtime failed before producing a valid result."],
            )
            raise PredictionUnavailableError(
                f"Prediction runtime failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _warnings(unknown_formula_tokens: list[str], *, route_count: int) -> list[str]:
        warnings = [
            "Model routes are candidates for experimental validation, not literature evidence.",
            "Relative route ranking weights are not calibrated experimental success probabilities.",
            "The model is conditioned only on formula; it does not incorporate furnace, pressure, atmosphere, or safety constraints.",
        ]
        if unknown_formula_tokens:
            warnings.append(
                "Formula contains tokens outside the input vocabulary: "
                + ", ".join(unknown_formula_tokens)
            )
        if route_count < 3:
            warnings.append(
                f"Model returned {route_count} valid unique route candidates, fewer than the requested maximum of 3."
            )
        return warnings
