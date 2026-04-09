"""initial tables

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2024-01-01 00:00:00.000000

This is the FIRST migration — creates user_profiles and food_checks tables
from scratch. Every subsequent schema change gets its own new revision file.

How this file was generated:
  alembic revision --autogenerate -m "initial tables"
  Alembic compared Base.metadata (our ORM models) against an empty DB
  and generated CREATE TABLE statements for both tables.

To apply:
  alembic upgrade head

To undo (drops both tables):
  alembic downgrade base
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# ── Revision metadata ─────────────────────────────────────────────────────────
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None    # None = this is the first migration
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Apply this migration: create user_profiles and food_checks tables.
    Called by: alembic upgrade head
    """

    # ── user_profiles table ───────────────────────────────────────────────────
    # Stores one row per app user (medical history + demographics)
    op.create_table(
        "user_profiles",
        sa.Column("id",             sa.Integer(),     nullable=False, autoincrement=True),
        sa.Column("name",           sa.String(100),   nullable=False),
        sa.Column("age",            sa.Integer(),     nullable=False),
        sa.Column("gender",         sa.String(20),    nullable=False),
        sa.Column("height_cm",      sa.Float(),       nullable=False),
        sa.Column("weight_kg",      sa.Float(),       nullable=False),
        # JSON columns store Python lists/dicts directly in PostgreSQL's JSONB type
        sa.Column("diseases",       sa.JSON(),        nullable=True),
        sa.Column("allergies",      sa.JSON(),        nullable=True),
        sa.Column("activity_level", sa.String(50),    nullable=True),
        sa.Column("dietary_pref",   sa.String(50),    nullable=True),
        sa.Column("created_at",     sa.DateTime(),    nullable=True),
        sa.Column("updated_at",     sa.DateTime(),    nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── food_checks table ─────────────────────────────────────────────────────
    # Stores one row per food check — the history log
    op.create_table(
        "food_checks",
        sa.Column("id",             sa.Integer(),     nullable=False, autoincrement=True),
        sa.Column("user_id",        sa.Integer(),     nullable=False),
        sa.Column("input_mode",     sa.String(20),    nullable=False),
        sa.Column("query",          sa.String(500),   nullable=False),
        sa.Column("food_found",     sa.String(200),   nullable=True),
        sa.Column("verdict",        sa.String(10),    nullable=False),
        sa.Column("score",          sa.Integer(),     nullable=False),
        sa.Column("warnings",       sa.JSON(),        nullable=True),
        sa.Column("reasons",        sa.JSON(),        nullable=True),
        sa.Column("nutrients",      sa.JSON(),        nullable=True),
        sa.Column("ml_prediction",  sa.String(20),    nullable=True),
        sa.Column("confidence",     sa.Float(),       nullable=True),
        sa.Column("checked_at",     sa.DateTime(),    nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user_profiles.id"],
            ondelete="CASCADE",    # deleting a user deletes all their checks
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    # user_id index: speeds up "get all checks for user X" queries
    op.create_index("ix_food_checks_user_id",   "food_checks", ["user_id"])
    # checked_at index: speeds up ORDER BY checked_at DESC in history queries
    op.create_index("ix_food_checks_checked_at", "food_checks", ["checked_at"])


def downgrade() -> None:
    """
    Undo this migration: drop both tables.
    Called by: alembic downgrade base
    WARNING: This deletes ALL data in both tables.
    """
    op.drop_index("ix_food_checks_checked_at", table_name="food_checks")
    op.drop_index("ix_food_checks_user_id",    table_name="food_checks")
    op.drop_table("food_checks")
    op.drop_table("user_profiles")
