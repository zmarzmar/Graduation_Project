"""create_initial_tables

Revision ID: 0001
Revises:
Create Date: 2026-03-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # papers 테이블
    op.create_table(
        "papers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("arxiv_id", sa.String(50), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("authors", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("abstract", sa.Text(), nullable=False, server_default=""),
        sa.Column("url", sa.Text(), nullable=False, server_default=""),
        sa.Column("pdf_url", sa.Text(), nullable=False, server_default=""),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("categories", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("citation_count", sa.Integer(), nullable=True),
        sa.Column("tldr", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("arxiv_id"),
    )
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"])

    # analysis_results 테이블
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.Integer(), nullable=True),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("query", sa.Text(), nullable=False, server_default=""),
        sa.Column("generated_code", sa.Text(), nullable=False, server_default=""),
        sa.Column("review_feedback", sa.Text(), nullable=False, server_default=""),
        sa.Column("review_passed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("iteration_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_results_paper_id", "analysis_results", ["paper_id"])

    # search_history 테이블
    op.create_table(
        "search_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("search_history")
    op.drop_index("ix_analysis_results_paper_id", table_name="analysis_results")
    op.drop_table("analysis_results")
    op.drop_index("ix_papers_arxiv_id", table_name="papers")
    op.drop_table("papers")
