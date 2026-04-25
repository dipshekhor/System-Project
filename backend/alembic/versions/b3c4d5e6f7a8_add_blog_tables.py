"""add blog tables

Revision ID: b3c4d5e6f7a8
Revises: 9da8df87730e
Create Date: 2026-04-25 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "9da8df87730e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "blog_posts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_blog_posts_user_id",    "blog_posts", ["user_id"])
    op.create_index("ix_blog_posts_created_at", "blog_posts", ["created_at"])

    op.create_table(
        "blog_answers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["post_id"], ["blog_posts.id"],    ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_blog_answers_post_id", "blog_answers", ["post_id"])
    op.create_index("ix_blog_answers_user_id", "blog_answers", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_blog_answers_user_id", table_name="blog_answers")
    op.drop_index("ix_blog_answers_post_id", table_name="blog_answers")
    op.drop_table("blog_answers")
    op.drop_index("ix_blog_posts_created_at", table_name="blog_posts")
    op.drop_index("ix_blog_posts_user_id",    table_name="blog_posts")
    op.drop_table("blog_posts")
