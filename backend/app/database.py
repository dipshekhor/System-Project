"""
database.py
===========
Day 10–11 · Database Connection & Session
-------------------------------------------
Sets up the async SQLAlchemy engine and session factory.

Why async?
  FastAPI is built on asyncio. If we use a synchronous DB driver (like psycopg2),
  every DB query BLOCKS the event loop — meaning no other requests can be served
  while one is waiting for the DB response. With asyncpg + AsyncSession, the
  event loop can serve other requests while waiting for PostgreSQL.

Key objects created here:
  - engine           → one global connection pool (re-used across requests)
  - AsyncSessionLocal→ factory for creating per-request sessions
  - Base             → parent class for all ORM table definitions in models.py
  - get_db()         → FastAPI dependency that yields a session per request

DATABASE_URL is read from environment variable (set in docker-compose.yml).
Falls back to a local PostgreSQL URL for running outside Docker.
"""

import os
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

# ─── Database URL ─────────────────────────────────────────────────────────────
# docker-compose sets: DATABASE_URL=postgresql+asyncpg://foodapp:foodapp123@postgres:5432/food_recommendation
# 'postgres' is the Docker service name — Docker DNS resolves it to the container IP.
# Outside Docker (local dev): change @postgres to @localhost
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://foodapp:foodapp123@localhost:5432/food_recommendation",
)

# ─── Engine ───────────────────────────────────────────────────────────────────
# create_async_engine sets up the connection pool.
# pool_pre_ping=True: runs a lightweight "SELECT 1" before each connection to
# detect and discard stale connections (prevents errors after DB restarts).
# echo=False: don't log every SQL query (set to True for debugging)
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    **(
        {
            # SQLite (used in tests) uses NullPool and does not accept
            # pool_size/max_overflow; keep options minimal for compatibility.
        }
        if DATABASE_URL.startswith("sqlite+")
        else {
            "pool_pre_ping": True,
            "pool_size": 10,      # max persistent connections in the pool
            "max_overflow": 20,   # extra connections allowed when pool is full
        }
    ),
)

# ─── Session factory ──────────────────────────────────────────────────────────
# async_sessionmaker creates new AsyncSession objects on demand.
# expire_on_commit=False: keeps ORM objects accessible after commit
# (otherwise SQLAlchemy would expire them to force a re-fetch — we don't need that)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ─── Base class for ORM models ────────────────────────────────────────────────
# All table classes in models.py inherit from this Base.
# Alembic uses Base.metadata to auto-detect schema changes.
class Base(DeclarativeBase):
    pass


# ─── FastAPI dependency ───────────────────────────────────────────────────────
async def get_db():
    """
    Yield a database session for the duration of one HTTP request.

    Usage in a route:
        @router.post("/profile")
        async def create_profile(data: ProfileCreate, db: AsyncSession = Depends(get_db)):
            ...

    How it works:
      - Creates a new AsyncSession for this request
      - Yields it to the route handler
      - Commits if the handler succeeds
      - Rolls back if the handler raises any exception
      - Closes the session either way (releases connection back to pool)

    This is the standard FastAPI pattern for per-request DB sessions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()   # commit all changes made in this request
        except Exception:
            await session.rollback() # undo partial changes on error
            raise                    # re-raise so FastAPI returns a 500 response
