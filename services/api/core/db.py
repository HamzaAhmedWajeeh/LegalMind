"""
core/db.py
==========
Async SQLAlchemy setup — engine, session factory, and base declarative class.

Design Pattern: Repository-ready setup.
- The async engine is created once at startup.
- `get_db_session()` is a FastAPI dependency that yields a scoped session
  and guarantees it's closed even on exceptions.
- All ORM models inherit from `Base`.

Usage (FastAPI route):
    from core.db import get_db_session
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.get("/example")
    async def example(db: AsyncSession = Depends(get_db_session)):
        result = await db.execute(select(Document))
        ...
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------
# pool_pre_ping=True: test connections before using them from the pool
# (handles stale connections after Postgres restarts)
engine = create_async_engine(
    settings.postgres_url,
    pool_pre_ping=True,
    pool_size=10,           # Max persistent connections
    max_overflow=20,        # Extra connections under load
    echo=(settings.environment == "development"),  # Log SQL in dev only
)

# ------------------------------------------------------------------
# Session Factory
# ------------------------------------------------------------------
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit (safer for async)
    autoflush=False,
    autocommit=False,
)

# ------------------------------------------------------------------
# Base class for all ORM models
# ------------------------------------------------------------------
class Base(DeclarativeBase):
    """
    All SQLAlchemy ORM models inherit from this class.
    Provides the metadata registry used by Alembic migrations.
    """
    pass


# ------------------------------------------------------------------
# FastAPI Dependency
# ------------------------------------------------------------------
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.
    Automatically commits on success, rolls back on exception, always closes.

    Usage:
        async def my_route(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ------------------------------------------------------------------
# Context manager for use outside FastAPI (Celery tasks, scripts)
# ------------------------------------------------------------------
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database access outside of FastAPI
    (e.g., Celery workers, CLI scripts, agent tasks).

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database transaction rolled back")
            raise
        finally:
            await session.close()


# ------------------------------------------------------------------
# Lifecycle helpers (called from FastAPI lifespan)
# ------------------------------------------------------------------
async def init_db() -> None:
    """Create all tables if they don't exist. Used in tests / local dev."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised")


async def close_db() -> None:
    """Dispose connection pool on application shutdown."""
    await engine.dispose()
    logger.info("Database connection pool closed")
