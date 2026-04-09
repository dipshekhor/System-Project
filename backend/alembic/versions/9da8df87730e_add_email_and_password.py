"""add email and password

Revision ID: 9da8df87730e
Revises: a1b2c3d4e5f6
Create Date: 2026-03-22 18:55:22.288202

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9da8df87730e"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add as nullable first so existing rows can be backfilled safely.
    op.add_column("user_profiles", sa.Column("email", sa.String(length=255), nullable=True))
    op.add_column("user_profiles", sa.Column("hashed_password", sa.String(length=255), nullable=True))

    # Ensure existing rows satisfy NOT NULL + unique email constraints.
    op.execute(
        """
        UPDATE user_profiles
        SET email = 'user' || id::text || '@placeholder.local'
        WHERE email IS NULL OR email = ''
        """
    )
    op.execute(
        """
        UPDATE user_profiles
        SET hashed_password = 'TEMP_PASSWORD_RESET_REQUIRED'
        WHERE hashed_password IS NULL OR hashed_password = ''
        """
    )

    op.alter_column("user_profiles", "email", nullable=False)
    op.alter_column("user_profiles", "hashed_password", nullable=False)
    op.create_index(op.f("ix_user_profiles_email"), "user_profiles", ["email"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_user_profiles_email"), table_name="user_profiles")
    op.drop_column("user_profiles", "hashed_password")
    op.drop_column("user_profiles", "email")
