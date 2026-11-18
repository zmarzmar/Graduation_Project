from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class QaMessage(Base):
    """분석 기록 한 건에 대한 논문 Q&A 한 번 (질문 + 그때의 응답 전체). 분석 기록을 지우면 함께 지운다 (crud/analysis)."""

    __tablename__ = "qa_messages"
    __table_args__ = (Index("ix_qa_messages_analysis_created", "analysis_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # /ask 응답 그대로 (answerable·answer·claims·citations·related_formulas·web_answer …) — 화면이 같은 모양으로 다시 그린다
    response: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
