from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class User(Base):
    """유저 테이블"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    # 프로필 표시용 (선택 입력)
    full_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    affiliation: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # 코드 생성 / 트렌드 브리핑 설정
    preferred_framework: Mapped[str | None] = mapped_column(String(20), nullable=True)
    preferred_categories: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    # 계정 상태
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
