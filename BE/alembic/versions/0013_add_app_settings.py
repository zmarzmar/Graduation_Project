"""add app_settings (AI 사용 중지 스위치)

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0013'
down_revision: Union[str, Sequence[str], None] = '0012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 관리자가 운영 중에 켜고 끄는 설정 — 재시작·배포 뒤에도 남는다. 행이 없으면 켜진 것으로 본다
    op.create_table(
        'app_settings',
        sa.Column('key', sa.String(length=50), primary_key=True),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('app_settings')
