"""
Urban Congestion Prediction API — application entry point.

Lifespan handles:
  - Structured logging setup
  - ML model load (graceful — 503 if unavailable, no crash)
  - Prometheus instrumentation
  - DB engine teardown on shutdown
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.v1.traffic import router as traffic_router
from src.core.config import get_settings
from src.db.session import engine, AsyncSessionLocal
from src.schemas.traffic import HealthResponse
from src.services.predictor import PredictorState, load_model

settings = get_settings()


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if settings.env == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}.get(
                settings.log_level.upper(), 20
            )
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Startup ──────────────────────────────────────────────────────────────
    _configure_logging()
    log.info("startup", env=settings.env, version=settings.app_version)

    # Load ML model — non-fatal on failure
    predictor_state: PredictorState = load_model(settings.model_path)
    app.state.predictor = predictor_state

    if not predictor_state.is_ready:
        log.warning(
            "model_unavailable",
            detail=predictor_state.load_error,
            hint="Deploy a trained model to enable /predict",
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("shutdown")
    await engine.dispose()


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Async REST API for real-time urban traffic congestion prediction. "
        "Ingests GPS telemetry, runs a Scikit-Learn model, and persists readings "
        "for continuous model retraining."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(traffic_router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health(request: Request) -> HealthResponse:
    from sqlalchemy import text

    db_ok = False
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        model_loaded=request.app.state.predictor.is_ready,
        db_reachable=db_ok,
        version=settings.app_version,
    )


# ── Prometheus — instrument AFTER all routes are registered ───────────────────
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health", "/docs", "/redoc", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics")
