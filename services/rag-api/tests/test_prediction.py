from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import settings
from src.prediction.bundle import PredictionModelBundle
from src.prediction.contracts import (
    FormulaValidation,
    PredictionExecutionRequest,
    PredictionModelInfo,
    PredictionUnavailableError,
    PredictionValidationError,
)
from src.prediction.repository import PredictionRunRepository
from src.prediction.runtime import PredictionRuntime
from src.prediction.service import PredictionService
from src.prediction.validation import validate_prediction_routes


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "growth-route-transformer" / "v2.0.0"


def _model_info() -> PredictionModelInfo:
    return PredictionModelInfo(
        model_id="growth-route-transformer",
        model_version="v2.0.0",
        artifact_digest="a" * 64,
        supported_methods=["Flux", "CVT"],
        parameter_count=6_614_099,
    )


def _valid_raw_routes() -> list[dict]:
    return [
        {
            "rank": 1,
            "display_probability": 1.0,
            "method": "Flux",
            "raw_reactants": [
                {"name": "Mn", "type": "raw", "r": None, "elements": ["Mn"]},
                {"name": "Ga", "type": "raw", "r": None, "elements": ["Ga"]},
                {"name": "N2", "type": "raw", "r": None, "elements": ["N"]},
            ],
            "adtv_reactants": [],
            "growth": {
                "T_s": {"token": "TEMP_BIN_80", "range_c": [750, 760]},
                "T_e": {"token": "TEMP_BIN_70", "range_c": [650, 660]},
                "dur": {"token": "DUR_BIN_20", "range_h": [100, 105]},
            },
            "element_coverage_ok": True,
        }
    ]


def test_model_bundle_rejects_checkpoint_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    checkpoint = root / "models" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-the-expected-checkpoint")
    for relative in (
        "src/predict.py",
        "src/library.py",
        "src/network.py",
        "features/input_vocab.json",
        "features/output_vocab.json",
        "lib/config/token_meta.json",
        "lib/config/bucket_config.json",
        "lib/rawLib/reactant_element_map.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "model_id": "test-model",
                "model_version": "v0",
                "checkpoint": "models/model.pth",
                "checkpoint_sha256": hashlib.sha256(b"different").hexdigest(),
                "task": {"supported_methods": ["Flux", "CVT"]},
            }
        ),
        encoding="utf-8",
    )

    bundle = PredictionModelBundle.load(root)
    with pytest.raises(PredictionUnavailableError, match="digest mismatch"):
        bundle.verify_checkpoint_digest()


def test_route_validation_requires_element_coverage() -> None:
    routes = validate_prediction_routes(
        _valid_raw_routes(),
        target_elements=["Mn", "Ga", "N"],
        model=_model_info(),
        requested_count=3,
    )
    assert routes[0].method == "Flux"
    assert routes[0].growth["T_s"]["range_c"] == [750, 760]

    invalid = _valid_raw_routes()
    invalid[0]["raw_reactants"] = invalid[0]["raw_reactants"][:-1]
    with pytest.raises(PredictionValidationError, match="coverage"):
        validate_prediction_routes(
            invalid,
            target_elements=["Mn", "Ga", "N"],
            model=_model_info(),
            requested_count=3,
        )


def test_prediction_run_repository_is_user_scoped(tmp_path: Path) -> None:
    repository = PredictionRunRepository(f"sqlite:///{tmp_path / 'prediction.sqlite3'}")
    model = _model_info()
    repository.register_model(
        model, {"model_id": model.model_id, "model_version": model.model_version}
    )
    run_id = repository.create_run(
        PredictionExecutionRequest(user_id="alice", formula="Mn3GaN"),
        formula_std="Mn3GaN",
        model=model,
    )
    repository.complete_run(
        run_id,
        result={"prediction_run_id": run_id, "routes": []},
        warnings=["candidate only"],
        runtime_ms=12,
    )

    alice_runs = repository.list_runs(user_id="alice")
    bob_runs = repository.list_runs(user_id="bob")
    assert [item["prediction_run_id"] for item in alice_runs] == [run_id]
    assert bob_runs == []
    assert alice_runs[0]["warnings"] == ["candidate only"]


class _FakeRuntime:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            model_info=_model_info(),
            manifest={"model_id": "growth-route-transformer", "model_version": "v2.0.0"},
        )
        self.generate_calls = 0

    def validate_formula(self, formula: str) -> FormulaValidation:
        return FormulaValidation(
            formula=formula,
            formula_std="Mn3GaN",
            formula_tokens=["Mn", "3", "Ga", "N"],
            target_elements=["Mn", "Ga", "N"],
            unknown_formula_tokens=[],
        )

    def generate(self, formula: str) -> dict:
        self.generate_calls += 1
        return {
            "formula": formula,
            "formula_std": "Mn3GaN",
            "formula_tokens": ["Mn", "3", "Ga", "N"],
            "target_elements": ["Mn", "Ga", "N"],
            "unknown_formula_tokens": [],
            "routes": _valid_raw_routes(),
        }


def test_prediction_service_persists_explicit_run(tmp_path: Path) -> None:
    config = replace(
        settings,
        prediction_database_url=f"sqlite:///{tmp_path / 'prediction.sqlite3'}",
        prediction_return_sequences=3,
    )
    repository = PredictionRunRepository(config.prediction_database_url)
    runtime = _FakeRuntime()
    service = PredictionService(config, runtime=runtime, repository=repository)  # type: ignore[arg-type]

    result = asyncio.run(
        service.predict(
            PredictionExecutionRequest(
                user_id="researcher-alice",
                session_id="session-1",
                formula="Mn3GaN",
            )
        )
    )

    assert runtime.generate_calls == 1
    assert result.source == "explicit_prediction"
    assert result.routes[0].element_coverage_ok is True
    saved = repository.list_runs(user_id="researcher-alice")
    assert saved[0]["status"] == "completed"
    assert saved[0]["result"]["prediction_run_id"] == result.prediction_run_id


@pytest.mark.integration
def test_real_prediction_runtime_smoke() -> None:
    runtime = PredictionRuntime(settings)
    formula = runtime.validate_formula("Mn3GaN")
    result = runtime.generate(formula.formula)
    routes = validate_prediction_routes(
        result["routes"],
        target_elements=result["target_elements"],
        model=runtime.bundle.model_info,
        requested_count=settings.prediction_return_sequences,
    )
    assert runtime.is_loaded
    assert result["formula_std"] == formula.formula_std
    assert 1 <= len(routes) <= 3
