from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from ..accounts.dependencies import require_current_account
from ..accounts.store import Account
from ..conversations.store import get_default_conversation_store
from ..prediction import (
    PredictionExecutionRequest,
    PredictionInputError,
    PredictionUnavailableError,
    PredictionValidationError,
    get_default_prediction_service,
)
from ..schemas.prediction import PredictionRequest, PredictionResponse


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    account: Account = Depends(require_current_account),
) -> dict:
    if request.session_id:
        session = await asyncio.to_thread(
            get_default_conversation_store().get_session,
            user_id=account.id,
            session_id=request.session_id,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
    service = get_default_prediction_service()
    try:
        result = await service.predict(
            PredictionExecutionRequest(
                user_id=account.id,
                session_id=request.session_id,
                message_id=request.message_id,
                formula=request.formula,
                source="explicit_prediction",
            )
        )
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionValidationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except PredictionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.as_dict()


@router.get("/prediction-runs")
async def list_prediction_runs(
    limit: int = Query(default=20, ge=1, le=100),
    account: Account = Depends(require_current_account),
) -> dict:
    service = get_default_prediction_service()
    runs = await asyncio.to_thread(service.repository.list_runs, user_id=account.id, limit=limit)
    return {"runs": runs}
