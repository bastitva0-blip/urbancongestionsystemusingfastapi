"""
Traffic prediction endpoints.
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import AuthKey, DBSession
from src.models.telemetry import TrafficHistory
from src.models.zone import Zone
from src.schemas.traffic import PredictRequest, PredictResponse, congestion_label
from src.services import predictor as pred_svc

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/traffic", tags=["Traffic"])


async def _lookup_zone(session: AsyncSession, lat: float, lng: float) -> Zone | None:
    """
    Use PostGIS ST_Contains to find the zone whose polygon contains the point.
    Returns the first match (zones should be non-overlapping).
    """
    point_wkt = f"SRID=4326;POINT({lng} {lat})"
    stmt = (
        select(Zone)
        .where(text(f"ST_Contains(boundary, ST_GeomFromEWKT('{point_wkt}'))"))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _persist_reading(
    session: AsyncSession,
    zone_id: int,
    reported_at: datetime,
    congestion_level: float,
) -> None:
    """Fire-and-forget DB write for retraining data accumulation."""
    try:
        record = TrafficHistory(
            reported_at=reported_at,
            zone_id=zone_id,
            congestion_level=congestion_level,
        )
        session.add(record)
        await session.commit()
    except Exception as exc:
        log.error("persist_failed", error=str(exc))
        await session.rollback()


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict congestion for a GPS coordinate",
)
async def predict_congestion(
    body: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: DBSession,
    _key: AuthKey,
) -> PredictResponse:
    # --- 1. Model availability check ---
    state = request.app.state.predictor
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {state.load_error}",
        )

    # --- 2. Zone lookup via PostGIS ---
    zone = await _lookup_zone(session, body.lat, body.lng)
    zone_id = zone.id if zone else 0  # 0 = "unknown zone" feature value

    # --- 3. Inference ---
    try:
        prob = pred_svc.predict(state, body.lat, body.lng, body.timestamp, zone_id)
    except Exception as exc:
        log.error("inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        ) from exc

    log.info(
        "prediction_made",
        lat=body.lat,
        lng=body.lng,
        zone_id=zone_id,
        probability=round(prob, 4),
    )

    # --- 4. Async persist (non-blocking) ---
    if zone:
        background_tasks.add_task(
            _persist_reading,
            session,
            zone.id,
            body.timestamp.replace(tzinfo=timezone.utc) if body.timestamp.tzinfo is None else body.timestamp,
            prob,
        )

    return PredictResponse(
        congestion_probability=prob,
        congestion_label=congestion_label(prob),
        zone_id=zone.id if zone else None,
        zone_name=zone.zone_name if zone else None,
        model_version=state.model_version,
    )


@router.get(
    "/zones",
    summary="List all defined traffic zones",
)
async def list_zones(session: DBSession, _key: AuthKey) -> list[dict]:
    result = await session.execute(select(Zone.id, Zone.zone_name))
    return [{"id": row.id, "zone_name": row.zone_name} for row in result.all()]
