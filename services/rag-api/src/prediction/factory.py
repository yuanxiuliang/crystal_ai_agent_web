from __future__ import annotations

import threading

from .service import PredictionService


_default_service: PredictionService | None = None
_default_service_lock = threading.Lock()


def get_default_prediction_service() -> PredictionService:
    global _default_service
    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                _default_service = PredictionService()
    return _default_service
