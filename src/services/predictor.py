"""
Predictor service.

Model is loaded ONCE at startup via FastAPI lifespan and stored in app.state.
If the model file is missing or corrupt the service continues running but
/predict returns HTTP 503 rather than crashing the process.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class PredictorState:
    model: Any = None
    model_version: str = "unavailable"
    is_ready: bool = False
    load_error: str = ""


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_path: str) -> PredictorState:
    path = Path(model_path)
    state = PredictorState()

    if not path.exists():
        msg = f"Model file not found: {path}"
        log.error("model_load_failed", reason=msg)
        state.load_error = msg
        return state

    try:
        model = joblib.load(path)
        version = _file_md5(path)
        log.info(
            "model_loaded",
            path=str(path),
            version=version,
            model_type=type(model).__name__,
        )
        # Log feature importances if available
        if hasattr(model, "feature_importances_"):
            features = ["hour", "day_of_week", "zone_id"]
            importances = dict(zip(features, model.feature_importances_))
            log.info("model_feature_importances", **importances)

        state.model = model
        state.model_version = version
        state.is_ready = True
    except Exception as exc:
        msg = f"Failed to deserialize model: {exc}"
        log.error("model_load_failed", reason=msg)
        state.load_error = msg

    return state


def extract_features(lat: float, lng: float, timestamp, zone_id: int) -> np.ndarray:
    """
    Feature vector: [hour, day_of_week, zone_id].

    hour        — 0-23, captures rush-hour signal
    day_of_week — 0 (Monday) to 6 (Sunday)
    zone_id     — encodes spatial context as a label
    """
    hour = timestamp.hour
    dow = timestamp.weekday()
    return np.array([[hour, dow, zone_id]])


def predict(state: PredictorState, lat: float, lng: float, timestamp, zone_id: int) -> float:
    """
    Returns congestion probability in [0, 1].
    Raises RuntimeError if model is not loaded.
    """
    if not state.is_ready:
        raise RuntimeError(state.load_error or "Model not loaded")

    features = extract_features(lat, lng, timestamp, zone_id)

    if hasattr(state.model, "predict_proba"):
        proba = state.model.predict_proba(features)[0]
        # Index 1 = "congested" class
        return float(proba[1])
    else:
        # Binary classifier without probability — return 0 or 1
        return float(state.model.predict(features)[0])
