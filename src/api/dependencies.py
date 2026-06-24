"""
Shared FastAPI dependencies injected via Depends().
"""

import hashlib
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.db.session import get_session

log = structlog.get_logger(__name__)
settings = get_settings()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)],
) -> str:
    """
    Validates the X-API-Key header against the stored SHA-256 hash.
    If API_KEY_HASH is empty (dev mode), authentication is skipped.
    """
    if not settings.api_key_hash:
        # Dev shortcut — warn loudly but don't block
        log.warning("auth_skipped", reason="API_KEY_HASH not configured")
        return "dev"

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if provided_hash != settings.api_key_hash:
        log.warning("auth_failed", provided_prefix=api_key[:6] + "...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


# Type alias for cleaner route signatures
DBSession = Annotated[AsyncSession, Depends(get_session)]
AuthKey = Annotated[str, Security(verify_api_key)]
