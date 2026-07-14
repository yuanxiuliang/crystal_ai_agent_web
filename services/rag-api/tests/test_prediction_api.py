from __future__ import annotations

from fastapi.testclient import TestClient

from src.accounts.dependencies import require_current_account
from src.accounts.store import Account
from src.api import prediction as prediction_api
from src.main import app
from src.prediction.contracts import PredictionModelInfo, PredictionResult, PredictionRoute


class _FakeRepository:
    def list_runs(self, *, user_id: str, limit: int) -> list[dict]:
        return [{"prediction_run_id": "prediction-test", "user_id": user_id, "limit": limit}]


class _FakePredictionService:
    repository = _FakeRepository()

    async def predict(self, request):
        return PredictionResult(
            prediction_run_id="prediction-test",
            source=request.source,
            formula=request.formula,
            formula_std=request.formula,
            formula_tokens=["Mn", "3", "Ga", "N"],
            target_elements=["Mn", "Ga", "N"],
            unknown_formula_tokens=[],
            routes=[
                PredictionRoute(
                    rank=1,
                    relative_rank_weight=1.0,
                    method="Flux",
                    raw_reactants=[],
                    additives=[],
                    growth={
                        "T_s": {"range_c": [750, 760]},
                        "T_e": {"range_c": [650, 660]},
                        "dur": None,
                    },
                    element_coverage_ok=True,
                )
            ],
            model=PredictionModelInfo(
                model_id="growth-route-transformer",
                model_version="v2.0.0",
                artifact_digest="a" * 64,
                supported_methods=["Flux", "CVT"],
                parameter_count=6_614_099,
            ),
            warnings=[],
            runtime_ms=1,
        )


def test_prediction_api_uses_prediction_service(monkeypatch) -> None:
    monkeypatch.setattr(
        prediction_api, "get_default_prediction_service", lambda: _FakePredictionService()
    )
    app.dependency_overrides[require_current_account] = lambda: Account(
        id="alice", email="alice@example.com", created_at="2026-01-01T00:00:00+00:00"
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/rag/predict",
            json={"formula": "Mn3GaN"},
        )
        runs = client.get("/api/rag/prediction-runs", params={"limit": 4})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prediction_run_id"] == "prediction-test"
    assert response.json()["source"] == "explicit_prediction"
    assert runs.status_code == 200
    assert runs.json()["runs"] == [
        {"prediction_run_id": "prediction-test", "user_id": "alice", "limit": 4}
    ]
