"""add researcher profile fields to users

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('bio', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('github_url', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('research_interests', sa.ARRAY(sa.String()), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'research_interests')
    op.drop_column('users', 'github_url')
    op.drop_column('users', 'bio')
