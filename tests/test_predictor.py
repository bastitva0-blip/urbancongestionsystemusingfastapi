"""Unit tests for the predictor service."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.services.predictor import (
    PredictorState,
    extract_features,
    load_model,
    predict,
)
from datetime import datetime


def _make_state() -> PredictorState:
    rng = np.random.default_rng(1)
    X = rng.integers(0, [24, 7, 10], size=(300, 3))
    y = np.isin(X[:, 0], [8, 17]).astype(int)
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)
    return PredictorState(model=clf, model_version="v-test", is_ready=True)


class TestExtractFeatures:
    def test_shape(self):
        ts = datetime(2024, 8, 12, 8, 0, 0)  # Monday 08:00
        feat = extract_features(40.71, -74.0, ts, zone_id=3)
        assert feat.shape == (1, 3)

    def test_hour_extraction(self):
        ts = datetime(2024, 8, 12, 17, 30, 0)
        feat = extract_features(0, 0, ts, zone_id=1)
        assert feat[0, 0] == 17  # hour

    def test_weekday(self):
        ts = datetime(2024, 8, 12, 12, 0, 0)  # Monday = 0
        feat = extract_features(0, 0, ts, zone_id=1)
        assert feat[0, 1] == 0


class TestPredict:
    def test_returns_float_in_range(self):
        state = _make_state()
        ts = datetime(2024, 8, 12, 8, 0, 0)
        prob = predict(state, 40.71, -74.0, ts, zone_id=1)
        assert 0.0 <= prob <= 1.0

    def test_raises_when_not_ready(self):
        state = PredictorState(is_ready=False, load_error="no model")
        ts = datetime(2024, 8, 12, 8, 0, 0)
        with pytest.raises(RuntimeError, match="no model"):
            predict(state, 40.71, -74.0, ts, zone_id=1)

    def test_rush_hour_higher_probability(self):
        state = _make_state()
        rush = datetime(2024, 8, 12, 8, 0, 0)    # Monday 8 AM
        quiet = datetime(2024, 8, 12, 3, 0, 0)   # Monday 3 AM
        p_rush = predict(state, 40.71, -74.0, rush, zone_id=1)
        p_quiet = predict(state, 40.71, -74.0, quiet, zone_id=1)
        assert p_rush > p_quiet


class TestLoadModel:
    def test_missing_file_returns_not_ready(self, tmp_path):
        state = load_model(str(tmp_path / "nonexistent.pkl"))
        assert not state.is_ready
        assert "not found" in state.load_error.lower()

    def test_corrupt_file_returns_not_ready(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        bad.write_bytes(b"this is not a pickle")
        state = load_model(str(bad))
        assert not state.is_ready
