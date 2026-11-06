"""add code_skip_reason to analysis_results

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0010'
down_revision: Union[str, Sequence[str], None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Coder가 구현 대상이 아니라고 판단한 이유 — 기존 기록은 모두 코드 생성을 시도했으므로 빈 값
    op.add_column('analysis_results', sa.Column('code_skip_reason', sa.Text(), nullable=False, server_default=''))


def downgrade() -> None:
    op.drop_column('analysis_results', 'code_skip_reason')
