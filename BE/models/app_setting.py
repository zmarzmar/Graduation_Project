from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class AppSetting(Base):
    """운영 중 관리자가 바꾸는 켜기/끄기 설정. 재시작·배포 뒤에도 남도록 DB에 둔다 (예: ai_enabled)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
