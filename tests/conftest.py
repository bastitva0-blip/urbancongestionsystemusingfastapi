"""
Test fixtures.

Uses real FastAPI app with:
  - In-memory trained model (no file I/O)
  - DB session fully mocked (no live Postgres needed for unit tests)
  - Monkey-patch for prometheus-fastapi-instrumentator/_IncludedRouter compat
    issue introduced in FastAPI 0.111+ where include_router adds an
    _IncludedRouter sentinel that lacks a .path attribute.
"""

import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import GradientBoostingClassifier

# ── Environment must be set before any src.* import ──────────────────────────
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test_user:test_secret@localhost:5433/test_congestion",
)
os.environ.setdefault("API_KEY_HASH", "")
os.environ.setdefault("MODEL_PATH", "./ml_pipeline/models/congestion_model.pkl")
os.environ.setdefault("ENV", "test")

# ── Compatibility shim ────────────────────────────────────────────────────────
# prometheus-fastapi-instrumentator iterates app.routes and calls route.path on
# every object.  FastAPI 0.111+ adds _IncludedRouter sentinel objects that have
# no .path attribute, crashing the middleware.  We patch the routing helper to
# skip objects without .path rather than raising AttributeError.
from prometheus_fastapi_instrumentator import routing as _pfi_routing

_original_get_route_name = _pfi_routing._get_route_name


def _safe_get_route_name(scope, routes, route_name=None):
    for route in routes:
        if not hasattr(route, "path"):
            continue
        try:
            match, child_scope = route.matches(scope)
        except Exception:
            continue
        from starlette.routing import Match, Mount
        if match == Match.FULL:
            route_name = route.path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child = _safe_get_route_name(child_scope, route.routes, route_name)
                route_name = None if child is None else route_name + child
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
    return None


_pfi_routing._get_route_name = _safe_get_route_name
# ─────────────────────────────────────────────────────────────────────────────

from src.services.predictor import PredictorState


def _make_tiny_model() -> GradientBoostingClassifier:
    rng = np.random.default_rng(0)
    X = rng.integers(0, [24, 7, 10], size=(300, 3))
    y = np.isin(X[:, 0], [8, 9, 17, 18]).astype(int)
    clf = GradientBoostingClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="session")
def trained_predictor() -> PredictorState:
    clf = _make_tiny_model()
    return PredictorState(model=clf, model_version="test-abc123", is_ready=True)


@pytest_asyncio.fixture
async def client(trained_predictor: PredictorState) -> AsyncIterator[AsyncClient]:
    """Async test client: model pre-loaded, DB calls mocked."""
    from src.main import app
    app.state.predictor = trained_predictor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
