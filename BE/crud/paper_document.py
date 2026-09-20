import hashlib
import json

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.analysis import AnalysisResult
from models.paper_document import PaperDocument

# pg_advisory_xact_lock(namespace, user_id)의 첫 번째 키 — 이 용도에만 쓰는 임의의 고정값
_DOCUMENT_LOCK_NAMESPACE = 7_310_001


def normalize_pages(pages: list[str]) -> list[str]:
    """저장 가능한 형태로 정리한다. Postgres text/JSON은 NUL 문자를 저장할 수 없는데 PDF 추출 텍스트에 섞여 나올 수 있다."""
    return [page.replace("\x00", "") for page in pages]


def hash_pages(pages: list[str]) -> str:
    """'실제로 저장하는' 페이지 목록의 sha256. 같은 논문이 아니라 '동일한 추출 본문'을 가리는 기준이다.

    JSON 직렬화를 해시해서 페이지 경계를 모호하지 않게 포함한다 — 본문에 어떤 문자가 있어도
    ["ab", "c"]와 ["a", "bc"]는 다른 문서다.
    """
    return hashlib.sha256(json.dumps(pages, ensure_ascii=False).encode("utf-8")).hexdigest()


async def _lock_user_documents(db: AsyncSession, user_id: int) -> None:
    """같은 사용자의 문서 저장과 정리를 트랜잭션 끝까지 직렬화한다.

    락이 없으면 '새 분석이 기존 문서를 찾아 연결하는 중'에 다른 요청이 마지막 기록을 지우며 그 문서를
    정리할 수 있다 — 아직 커밋되지 않은 새 기록은 정리 쿼리에 보이지 않아서 문서가 지워지고,
    새 기록은 조용히 원문 없는 기록이 된다.
    """
    await db.execute(select(func.pg_advisory_xact_lock(_DOCUMENT_LOCK_NAMESPACE, user_id)))


async def get_or_create_document(
    db: AsyncSession,
    user_id: int,
    pages: list[str],
    source: str,
    title: str,
    arxiv_id: str | None = None,
) -> int:
    """사용자의 문서를 저장하고 id를 반환한다. 같은 사용자가 같은 본문을 다시 분석하면 기존 문서를 재사용한다.

    재사용할 때 기존 문서의 title·source는 덮어쓰지 않는다 — 분석별 정보(파일명·검색어·논문)는 분석 기록에 있다.
    호출한 트랜잭션이 끝날 때까지 사용자 문서 락을 쥐므로, 반환한 문서는 분석 기록을 연결해 커밋할 때까지 지워지지 않는다.
    """
    await _lock_user_documents(db, user_id)
    pages = normalize_pages(pages)
    doc_hash = hash_pages(pages)

    # 락과 별개로 (user_id, doc_hash) 유니크 제약이 최종 방어선이다.
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
    지운 문서의 (id, doc_hash)를 반환한다.
    """
    await _lock_user_documents(db, user_id)
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
