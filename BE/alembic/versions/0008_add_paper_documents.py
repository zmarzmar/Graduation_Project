"""add paper_documents and analysis_results.document_id

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0008'
down_revision: Union[str, Sequence[str], None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 분석에 사용한 논문 원문(페이지별) — 논문 Q&A(RAG)의 원본
    op.create_table(
        'paper_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('doc_hash', sa.String(length=64), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=False),
        sa.Column('arxiv_id', sa.String(length=50), nullable=True),
        sa.Column('title', sa.Text(), nullable=False, server_default=''),
        sa.Column('page_count', sa.Integer(), nullable=False),
        sa.Column('pages', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'doc_hash', name='uq_paper_documents_user_hash'),
    )
    op.create_index('ix_paper_documents_user_id', 'paper_documents', ['user_id'])

    # 기존 분석 기록은 원문이 없으므로 NULL로 남는다 (Q&A를 쓰려면 재분석 필요).
    # 문서가 지워져도 분석 기록은 남기고 연결만 해제한다.
    op.add_column('analysis_results', sa.Column('document_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_analysis_results_document_id', 'analysis_results', 'paper_documents',
        ['document_id'], ['id'], ondelete='SET NULL',
    )
    op.create_index('ix_analysis_results_document_id', 'analysis_results', ['document_id'])


def downgrade() -> None:
    op.drop_index('ix_analysis_results_document_id', table_name='analysis_results')
    op.drop_constraint('fk_analysis_results_document_id', 'analysis_results', type_='foreignkey')
    op.drop_column('analysis_results', 'document_id')
    op.drop_index('ix_paper_documents_user_id', table_name='paper_documents')
    op.drop_table('paper_documents')
