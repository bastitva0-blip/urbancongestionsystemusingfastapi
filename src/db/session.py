from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from src.core.config import get_settings

settings = get_settings()

# NullPool is used for compatibility with pytest-asyncio (no shared connections across tests)
engine = create_async_engine(
    settings.database_url,
    echo=(settings.env == "development"),
    poolclass=NullPool if settings.env == "test" else None,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
