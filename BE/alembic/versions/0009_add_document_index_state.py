"""add index job state and purge tombstone to paper_documents

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-20 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0009'
down_revision: Union[str, Sequence[str], None] = '0008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 벡터 색인 상태 — 기존 문서는 아직 색인되지 않았으므로 'none'
    op.add_column('paper_documents', sa.Column('index_status', sa.String(length=20), nullable=False, server_default='none'))
    op.add_column('paper_documents', sa.Column('index_job_id', sa.String(length=32), nullable=True))
    op.add_column('paper_documents', sa.Column('index_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('paper_documents', sa.Column('indexed_chunk_count', sa.Integer(), nullable=True))
    op.add_column('paper_documents', sa.Column('index_error', sa.Text(), nullable=True))
    # 삭제 툼스톤 — 벡터 색인 삭제가 끝날 때까지 재시도에 필요한 행을 남겨둔다
    op.add_column('paper_documents', sa.Column('purge_pending_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_paper_documents_purge_pending_at', 'paper_documents', ['purge_pending_at'])


def downgrade() -> None:
    op.drop_index('ix_paper_documents_purge_pending_at', table_name='paper_documents')
    for column in ('purge_pending_at', 'index_error', 'indexed_chunk_count', 'index_started_at', 'index_job_id', 'index_status'):
        op.drop_column('paper_documents', column)
