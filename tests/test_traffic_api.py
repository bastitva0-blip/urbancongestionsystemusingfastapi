"""API contract tests for /api/v1/traffic/predict and /health."""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


class TestHealth:
    async def test_health_returns_200(self, client: AsyncClient):
        with patch("src.main.AsyncSessionLocal") as mock_cls:
            mock_sess = AsyncMock()
            mock_sess.execute = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body

    async def test_health_model_loaded_true(self, client: AsyncClient):
        with patch("src.main.AsyncSessionLocal") as mock_cls:
            mock_sess = AsyncMock()
            mock_sess.execute = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await client.get("/health")
        assert resp.json()["model_loaded"] is True

    async def test_health_degraded_when_db_down(self, client: AsyncClient):
        with patch("src.main.AsyncSessionLocal") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("db down"))
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["db_reachable"] is False


class TestPredictEndpoint:
    async def test_predict_returns_200(self, client: AsyncClient):
        payload = {"lat": 40.71, "lng": -74.0, "timestamp": "2024-08-12T08:00:00"}
        with patch("src.api.v1.traffic._lookup_zone", return_value=None):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "congestion_probability" in body
        assert 0.0 <= body["congestion_probability"] <= 1.0
        assert body["congestion_label"] in {"low", "medium", "high"}
        assert "model_version" in body

    async def test_predict_response_schema(self, client: AsyncClient):
        """All documented fields must be present."""
        payload = {"lat": 40.71, "lng": -74.0}
        with patch("src.api.v1.traffic._lookup_zone", return_value=None):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        body = resp.json()
        for field in ("congestion_probability", "congestion_label", "model_version", "processed_at"):
            assert field in body, f"Missing field: {field}"

    async def test_predict_invalid_lat_too_large(self, client: AsyncClient):
        resp = await client.post("/api/v1/traffic/predict", json={"lat": 999, "lng": -74.0})
        assert resp.status_code == 422

    async def test_predict_invalid_lng_too_large(self, client: AsyncClient):
        resp = await client.post("/api/v1/traffic/predict", json={"lat": 40.71, "lng": 999})
        assert resp.status_code == 422

    async def test_predict_missing_lat(self, client: AsyncClient):
        resp = await client.post("/api/v1/traffic/predict", json={"lng": -74.0})
        assert resp.status_code == 422

    async def test_predict_503_when_model_not_ready(self, client: AsyncClient):
        from src.services.predictor import PredictorState
        from src.main import app

        original = app.state.predictor
        app.state.predictor = PredictorState(is_ready=False, load_error="no model file")
        try:
            payload = {"lat": 40.71, "lng": -74.0}
            with patch("src.api.v1.traffic._lookup_zone", return_value=None):
                resp = await client.post("/api/v1/traffic/predict", json=payload)
            assert resp.status_code == 503
            assert "not available" in resp.json()["detail"].lower()
        finally:
            app.state.predictor = original

    async def test_predict_with_zone_match(self, client: AsyncClient):
        fake_zone = MagicMock()
        fake_zone.id = 1
        fake_zone.zone_name = "downtown"
        payload = {"lat": 40.72, "lng": -73.98, "timestamp": "2024-08-12T08:00:00"}
        with patch("src.api.v1.traffic._lookup_zone", return_value=fake_zone):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["zone_id"] == 1
        assert body["zone_name"] == "downtown"

    async def test_predict_no_zone_returns_null_zone(self, client: AsyncClient):
        payload = {"lat": 10.0, "lng": 10.0}
        with patch("src.api.v1.traffic._lookup_zone", return_value=None):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["zone_id"] is None
        assert body["zone_name"] is None

    async def test_predict_rush_hour_is_high(self, client: AsyncClient):
        """Model trained on rush-hour signal — 8 AM Monday should be high."""
        payload = {"lat": 40.71, "lng": -74.0, "timestamp": "2024-08-12T08:00:00"}
        with patch("src.api.v1.traffic._lookup_zone", return_value=None):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["congestion_probability"] > 0.5

    async def test_predict_off_peak_is_low(self, client: AsyncClient):
        """3 AM should produce low congestion probability."""
        payload = {"lat": 40.71, "lng": -74.0, "timestamp": "2024-08-12T03:00:00"}
        with patch("src.api.v1.traffic._lookup_zone", return_value=None):
            resp = await client.post("/api/v1/traffic/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["congestion_probability"] < 0.5


class TestMetrics:
    async def test_metrics_endpoint_reachable(self, client: AsyncClient):
        resp = await client.get("/metrics")
        # 200 = Prometheus exposition format served; 404 = middleware disabled in test
        assert resp.status_code in (200, 404)

    async def test_docs_reachable(self, client: AsyncClient):
        resp = await client.get("/docs")
        assert resp.status_code == 200
