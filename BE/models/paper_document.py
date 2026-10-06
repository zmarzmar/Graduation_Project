from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class PaperDocument(Base):
    """분석에 사용한 논문 원문(페이지별) 테이블 — 논문 Q&A(RAG)의 원본.

    벡터 색인(Chroma)은 이 테이블의 pages에서 언제든 다시 만들 수 있는 파생 데이터다.
    접근 권한은 여기서 판정한다: 사용자는 자신의 삭제되지 않은 분석 기록이 가리키는 문서만 볼 수 있다.
    """

    __tablename__ = "paper_documents"
    __table_args__ = (
        # 같은 사용자가 같은 본문을 다시 분석해도 문서는 하나 — 사용자 간에는 공유하지 않는다
        UniqueConstraint("user_id", "doc_hash", name="uq_paper_documents_user_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 추출된 페이지 본문의 sha256 — '동일한 본문'을 중복 제거하는 기준 (같은 논문의 다른 판본은 다른 문서)
    doc_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # arxiv | upload
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # 페이지별 텍스트 — 빈 페이지 포함, 인덱스 + 1 = 원본 PDF 페이지 번호
    pages: Mapped[list] = mapped_column(JSON, nullable=False)

    # ── 벡터 색인 상태 (첫 질문 때 색인) ──────────────────────────────────
    # none | indexing | ready | failed. 검색은 ready일 때만, index_job_id의 청크만 대상으로 한다.
    index_status: Mapped[str] = mapped_column(String(20), nullable=False, default="none", server_default="none")
    # 색인을 선점할 때마다 새로 발급한다. 완료·실패 기록은 이 값이 일치하는 작업만 할 수 있어서,
    # 느리게 돌던 오래된 작업이 새 색인의 상태를 바꾸거나 새 색인을 대체할 수 없다.
    index_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    index_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    indexed_chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── 삭제 ─────────────────────────────────────────────────────────────
    # 마지막 활성 분석 기록이 지워지면 행을 바로 지우지 않고 툼스톤으로 만든다: 원문(pages)은 즉시 비우고
    # doc_hash를 바꿔 같은 본문의 새 문서 생성을 막지 않는다. 벡터 색인 삭제가 끝난 뒤에야 행을 지운다 —
    # Postgres와 Chroma는 한 트랜잭션이 아니므로 재시도에 필요한 id를 여기에 남겨둔다.
    purge_pending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
