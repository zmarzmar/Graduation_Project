"""change preferred_framework to array

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 기존 단일 문자열 → 배열로 변환 (기존 값은 단일 원소 배열로 이전)
    op.execute("""
        ALTER TABLE users
        ALTER COLUMN preferred_framework TYPE TEXT[]
        USING CASE
            WHEN preferred_framework IS NULL THEN NULL
            ELSE ARRAY[preferred_framework]
        END
    """)


def downgrade() -> None:
    # 배열 → 단일 문자열 (첫 번째 원소만 보존)
    op.execute("""
        ALTER TABLE users
        ALTER COLUMN preferred_framework TYPE VARCHAR(20)
        USING CASE
            WHEN preferred_framework IS NULL THEN NULL
            ELSE preferred_framework[1]
        END
    """)
