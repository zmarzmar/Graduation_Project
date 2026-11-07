"""add exec_status, exec_detail, run_error to analysis_results

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0011'
down_revision: Union[str, Sequence[str], None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 생성 코드 실행 검사 결과 — 기존 기록은 실행 검사를 하지 않았으므로 빈 값
    op.add_column('analysis_results', sa.Column('exec_status', sa.String(length=20), nullable=False, server_default=''))
    op.add_column('analysis_results', sa.Column('exec_detail', sa.Text(), nullable=False, server_default=''))
    # LLM 호출 오류로 끝난 분석의 사유 — 기존 기록에는 남아 있지 않다
    op.add_column('analysis_results', sa.Column('run_error', sa.Text(), nullable=False, server_default=''))


def downgrade() -> None:
    op.drop_column('analysis_results', 'run_error')
    op.drop_column('analysis_results', 'exec_detail')
    op.drop_column('analysis_results', 'exec_status')
