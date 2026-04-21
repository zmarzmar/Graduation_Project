"""add_user_id_to_history

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-21

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "search_history",
        sa.Column("user_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_search_history_user_id",
        "search_history",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_search_history_user_id", "search_history", ["user_id"])

    op.add_column(
        "analysis_results",
        sa.Column("user_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_analysis_results_user_id",
        "analysis_results",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_analysis_results_user_id", "analysis_results", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_results_user_id", "analysis_results")
    op.drop_constraint("fk_analysis_results_user_id", "analysis_results", type_="foreignkey")
    op.drop_column("analysis_results", "user_id")

    op.drop_index("ix_search_history_user_id", "search_history")
    op.drop_constraint("fk_search_history_user_id", "search_history", type_="foreignkey")
    op.drop_column("search_history", "user_id")
