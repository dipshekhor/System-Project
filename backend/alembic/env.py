"""
alembic/env.py
==============
Alembic environment configuration.

This file is loaded by every alembic command.
The two most important things it does:

  1. Tells Alembic WHERE to find our ORM models
     (so --autogenerate can compare them against the live DB schema)

  2. Tells Alembic which DATABASE_URL to connect to
     (reads from the same DATABASE_URL env var as FastAPI)

Why we import models here:
  Alembic's autogenerate feature compares:
    - What's in Base.metadata (your ORM model definitions)
    - What's currently in the database
  And generates an ALTER TABLE / CREATE TABLE script for the diff.
  
  If you DON'T import your models here, Base.metadata is empty,
  and autogenerate produces an empty migration (no-op).

  The 'import app.models' line is what registers UserProfile
  and FoodCheck into Base.metadata.

Async note:
  Our FastAPI app uses asyncpg (async PostgreSQL driver).
  But Alembic's run_migrations_online() is synchronous.
  We handle this with run_async_migrations() + asyncio.run().
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ── Load alembic.ini logging config ──────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import ORM models so Alembic can detect schema ────────────────────────────
# CRITICAL: Both lines below MUST be here.
# Line 1: imports Base (the DeclarativeBase that all models inherit from)
# Line 2: imports the actual model classes — this registers them in Base.metadata
#          Without this line, autogenerate produces empty migrations.
import sys
from pathlib import Path
# Add backend/ to path so 'from app.database import Base' works
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import Base       # Base.metadata is what Alembic inspects
import app.models                   # noqa: F401 — registers UserProfile, FoodCheck

target_metadata = Base.metadata     # Alembic compares this vs the live DB


# ── Database URL ──────────────────────────────────────────────────────────────
# Use the same DATABASE_URL as FastAPI (set in docker-compose.yml environment).
# Fall back to the alembic.ini value for running outside Docker.
def get_url() -> str:
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    # asyncpg URL uses 'postgresql+asyncpg://' but Alembic's sync runner
    # needs 'postgresql+psycopg2://' — swap the driver for Alembic only.
    # Alembic uses psycopg2 internally; FastAPI uses asyncpg.
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


# ── Offline mode (generate SQL without connecting to DB) ─────────────────────
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates the migration SQL script to stdout/file WITHOUT connecting to DB.
    Useful for reviewing what SQL will run before applying it.
    Usage: alembic upgrade head --sql
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (connect to DB and apply migrations) ─────────────────────────
def do_run_migrations(connection: Connection) -> None:
    """Run migrations using an existing connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # compare_type=True: detect column type changes (e.g. String(50) → String(100))
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations using the async engine.
    We use run_sync() to bridge the sync Alembic API with the async engine.
    """
    connectable = create_async_engine(
        get_url().replace("postgresql+psycopg2://", "postgresql+asyncpg://"),
        poolclass=pool.NullPool,   # don't pool connections during migration
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode."""
    asyncio.run(run_async_migrations())


# ── Run ────────────────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
