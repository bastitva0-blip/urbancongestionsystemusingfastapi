from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in WGS84")
    lng: float = Field(..., ge=-180.0, le=180.0, description="Longitude in WGS84")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the GPS reading",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "lat": 40.7128,
            "lng": -74.0060,
            "timestamp": "2024-08-17T08:30:00Z",
        }
    }}


class PredictResponse(BaseModel):
    congestion_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted congestion probability (0–1)"
    )
    congestion_label: str = Field(
        ..., description="Human-readable label: 'low', 'medium', or 'high'"
    )
    zone_id: int | None = Field(
        None, description="Matched zone ID; null if coordinates fall outside all zones"
    )
    zone_name: str | None = Field(None, description="Matched zone name")
    model_version: str = Field(..., description="MD5 hash of the loaded model file")
    processed_at: datetime = Field(default_factory=_utcnow)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_reachable: bool
    version: str


def congestion_label(prob: float) -> str:
    if prob < 0.4:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"
