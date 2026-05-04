"""add_admin_user

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-04

"""
from typing import Sequence, Union

import bcrypt
import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="false"),
    )

    password_hash = bcrypt.hashpw("qwer1234".encode(), bcrypt.gensalt()).decode()
    bind = op.get_bind()
    existing = bind.execute(
        sa.text("SELECT id FROM users WHERE username = :username"),
        {"username": "admin"},
    ).first()

    if existing:
        bind.execute(
            sa.text(
                """
                UPDATE users
                SET password_hash = :password_hash,
                    email = :email,
                    full_name = :full_name,
                    is_active = true,
                    is_admin = true
                WHERE username = :username
                """
            ),
            {
                "username": "admin",
                "email": "admin@paperpilot.local",
                "full_name": "Administrator",
                "password_hash": password_hash,
            },
        )
        return

    bind.execute(
        sa.text(
            """
            INSERT INTO users (
                email,
                password_hash,
                username,
                full_name,
                preferred_framework,
                preferred_categories,
                is_active,
                is_admin
            )
            VALUES (
                :email,
                :password_hash,
                :username,
                :full_name,
                :preferred_framework,
                ARRAY[]::varchar[],
                true,
                true
            )
            """
        ),
        {
            "email": "admin@paperpilot.local",
            "password_hash": password_hash,
            "username": "admin",
            "full_name": "Administrator",
            "preferred_framework": "pytorch",
        },
    )


def downgrade() -> None:
    op.execute("DELETE FROM users WHERE username = 'admin' AND email = 'admin@paperpilot.local'")
    op.drop_column("users", "is_admin")
