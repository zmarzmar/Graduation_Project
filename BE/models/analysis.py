from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class AnalysisResult(Base):
    """에이전트 분석 결과 테이블"""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 연관 논문 (없을 수도 있음 — PDF 업로드 시 arxiv_id 없는 경우)
    paper_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("papers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # 분석에 사용한 원문 (논문 Q&A용). 게스트·초록 기반 분석·이 컬럼 추가 전의 기록은 NULL.
    # 문서가 지워지면 분석 기록은 남기고 연결만 해제한다 (소프트 삭제된 기록 포함).
    document_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("paper_documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # pdf | search | trend
    query: Mapped[str] = mapped_column(Text, nullable=False, default="")
    generated_code: Mapped[str] = mapped_column(Text, nullable=False, default="")
    review_feedback: Mapped[str] = mapped_column(Text, nullable=False, default="")
    review_passed: Mapped[bool] = mapped_column(nullable=False, default=False)
    iteration_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    paper_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    paper_review: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    key_formulas: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
