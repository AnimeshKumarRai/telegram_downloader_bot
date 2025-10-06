from __future__ import annotations

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import get_settings
from logger import setup_logging

logger = setup_logging()
_settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(_settings.postgres_dsn, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db():
    # Ensure models are imported so Base.metadata has all tables
    from models import User, CacheItem, JobLog  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def healthcheck_db() -> bool:
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("DB healthcheck failed: {}", e)
        return False