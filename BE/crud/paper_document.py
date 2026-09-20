import hashlib

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.analysis import AnalysisResult
from models.paper_document import PaperDocument


def hash_pages(pages: list[str]) -> str:
    """페이지 본문의 sha256. '동일한 추출 본문'을 가리는 기준이다.

    페이지 경계까지 포함해 해시한다 (폼피드로 구분) — 같은 글자라도 페이지 구성이 다르면 다른 문서.
    """
    return hashlib.sha256("\f".join(pages).encode("utf-8")).hexdigest()


async def get_or_create_document(
    db: AsyncSession,
    user_id: int,
    pages: list[str],
    source: str,
    title: str,
    arxiv_id: str | None = None,
) -> int:
    """사용자의 문서를 저장하고 id를 반환한다. 같은 사용자가 같은 본문을 다시 분석하면 기존 문서를 재사용한다."""
    # Postgres text/JSON은 NUL 문자를 저장할 수 없다 — PDF 추출 텍스트에 섞여 나오는 경우가 있다.
    pages = [page.replace("\x00", "") for page in pages]
    doc_hash = hash_pages(pages)

    # 동시에 들어온 같은 문서의 저장 요청은 유니크 제약으로 하나만 남긴다.
    inserted = await db.execute(
        insert(PaperDocument)
        .values(
            user_id=user_id, doc_hash=doc_hash, source=source, arxiv_id=arxiv_id,
            title=title, page_count=len(pages), pages=pages,
        )
        .on_conflict_do_nothing(constraint="uq_paper_documents_user_hash")
        .returning(PaperDocument.id)
    )
    if (document_id := inserted.scalar_one_or_none()) is not None:
        return document_id

    existing = await db.execute(
        select(PaperDocument.id).where(PaperDocument.user_id == user_id, PaperDocument.doc_hash == doc_hash)
    )
    return existing.scalar_one()


async def purge_unreferenced_documents(db: AsyncSession, user_id: int) -> list[tuple[int, str]]:
    """사용자의 '삭제되지 않은' 분석 기록이 하나도 가리키지 않는 문서를 지운다.

    기록 하나를 지워도 같은 문서를 쓰는 다른 기록이 남아 있으면 문서는 유지된다.
    분석 기록 행은 지우지 않는다 — FK의 ON DELETE SET NULL이 소프트 삭제된 기록까지 연결만 해제한다.
    지운 문서의 (id, doc_hash)를 반환한다 (파생 색인 정리에 사용).
    """
    still_referenced = select(AnalysisResult.document_id).where(
        AnalysisResult.user_id == user_id,
        AnalysisResult.is_deleted == False,  # noqa: E712
        AnalysisResult.document_id.is_not(None),
    )
    purged = await db.execute(
        delete(PaperDocument)
        .where(PaperDocument.user_id == user_id, PaperDocument.id.not_in(still_referenced))
        .returning(PaperDocument.id, PaperDocument.doc_hash)
    )
    return [(row.id, row.doc_hash) for row in purged.all()]
