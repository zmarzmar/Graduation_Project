"""add summary_claims to analysis_results

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0012'
down_revision: Union[str, Sequence[str], None] = '0011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 요약 문장별 원문 인용 — 기존 기록에는 없으므로 빈 목록 (화면은 paper_summary를 그대로 보여준다)
    op.add_column('analysis_results', sa.Column('summary_claims', sa.JSON(), nullable=False, server_default='[]'))


def downgrade() -> None:
    op.drop_column('analysis_results', 'summary_claims')
