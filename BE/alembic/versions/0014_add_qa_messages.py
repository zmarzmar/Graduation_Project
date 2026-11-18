"""add qa_messages (논문 Q&A 기록 저장)

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0014'
down_revision: Union[str, Sequence[str], None] = '0013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 분석 기록별 Q&A — 마이페이지에서 다시 보고 이어서 묻는다. 분석 기록 삭제(소프트) 때 crud가 함께 지운다
    op.create_table(
        'qa_messages',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('analysis_id', sa.Integer(), sa.ForeignKey('analysis_results.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('response', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_qa_messages_analysis_created', 'qa_messages', ['analysis_id', 'created_at'])
    op.create_index('ix_qa_messages_user_id', 'qa_messages', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_qa_messages_user_id', table_name='qa_messages')
    op.drop_index('ix_qa_messages_analysis_created', table_name='qa_messages')
    op.drop_table('qa_messages')
