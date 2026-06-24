from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://congestion:congestion_secret@localhost:5432/congestion_db"

    # API Security — store SHA-256 hash, never plaintext key
    api_key_hash: str = ""

    # Rate limiting
    rate_limit: str = "100/minute"

    # ML model
    model_path: str = "./ml_pipeline/models/congestion_model.pkl"

    # Logging
    log_level: str = "INFO"

    # App
    env: str = "production"
    app_title: str = "Urban Congestion Prediction API"
    app_version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
