import hashlib
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import String, and_, cast, delete, func, or_, select, update
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


async def mark_unreferenced_documents_for_purge(db: AsyncSession, user_id: int) -> list[int]:
    """사용자의 '삭제되지 않은' 분석 기록이 하나도 가리키지 않는 문서를 툼스톤으로 만든다. 툼스톤이 된 id를 반환한다.

    기록 하나를 지워도 같은 문서를 쓰는 다른 기록이 남아 있으면 문서는 유지된다.

    행을 바로 지우지 않는 이유: 벡터 색인(Chroma) 삭제는 이 트랜잭션에 포함되지 않아 따로 실패할 수 있다.
    재시도에 필요한 id를 DB에 남기되, 툼스톤으로 만드는 순간
    - pages를 비운다 → 원문은 이 커밋과 함께 사라진다
    - index_job_id를 지운다 → 진행 중이던 색인 작업은 더 이상 완료를 기록할 수 없다
    - doc_hash를 바꾼다 → 같은 본문을 다시 분석하면 '새 id의 새 문서'가 만들어진다.
      삭제 중인 문서는 복구하지 않는다 — 진행 중인 정리 작업이 되살린 문서를 지우는 충돌을 피한다.
    활성 분석 기록은 툼스톤을 가리키지 않으므로(가리키는 활성 기록이 없어야 툼스톤이 된다) 접근은 즉시 막힌다.
    """
    await _lock_user_documents(db, user_id)
    still_referenced = select(AnalysisResult.document_id).where(
        AnalysisResult.user_id == user_id,
        AnalysisResult.is_deleted == False,  # noqa: E712
        AnalysisResult.document_id.is_not(None),
    )
    marked = await db.execute(
        update(PaperDocument)
        .where(
            PaperDocument.user_id == user_id,
            PaperDocument.purge_pending_at.is_(None),
            PaperDocument.id.not_in(still_referenced),
        )
        .values(
            purge_pending_at=func.now(),
            pages=[],
            doc_hash=func.concat("purged:", cast(PaperDocument.id, String)),
            index_status="none",
            index_job_id=None,
            indexed_chunk_count=None,
        )
        .returning(PaperDocument.id)
    )
    return list(marked.scalars())


async def list_purge_pending(db: AsyncSession) -> list[tuple[int, datetime]]:
    """정리할 툼스톤의 (id, purge_pending_at) 목록."""
    rows = await db.execute(
        select(PaperDocument.id, PaperDocument.purge_pending_at).where(PaperDocument.purge_pending_at.is_not(None))
    )
    return [(row.id, row.purge_pending_at) for row in rows.all()]


async def list_live_index_jobs(db: AsyncSession) -> dict[int, str | None]:
    """삭제 대기가 아닌 모든 문서의 {document_id: 현재 index_job_id}. 벡터 색인의 고아 청크를 가려내는 기준이다."""
    rows = await db.execute(
        select(PaperDocument.id, PaperDocument.index_job_id).where(PaperDocument.purge_pending_at.is_(None))
    )
    return {row.id: row.index_job_id for row in rows.all()}


async def delete_purged_document(db: AsyncSession, document_id: int) -> None:
    """벡터 색인 삭제가 끝난 툼스톤 행을 지운다. 툼스톤이 아닌 문서는 절대 지우지 않는다.

    분석 기록 행은 남는다 — FK의 ON DELETE SET NULL이 소프트 삭제된 기록까지 연결만 해제한다.
    """
    await db.execute(
        delete(PaperDocument).where(PaperDocument.id == document_id, PaperDocument.purge_pending_at.is_not(None))
    )


async def get_document_for_analysis(db: AsyncSession, analysis_id: int, user_id: int) -> PaperDocument | None:
    """분석 기록을 통해 문서에 접근한다 — 접근 권한 판정은 여기 한 곳에서만 한다.

    본인 소유이고 삭제되지 않은 분석 기록이 가리키는, 삭제 중이 아닌 문서만 반환한다.
    """
    row = await db.execute(
        select(PaperDocument)
        .join(AnalysisResult, AnalysisResult.document_id == PaperDocument.id)
        .where(
            AnalysisResult.id == analysis_id,
            AnalysisResult.user_id == user_id,
            AnalysisResult.is_deleted == False,  # noqa: E712
            PaperDocument.user_id == user_id,
            PaperDocument.purge_pending_at.is_(None),
        )
    )
    return row.scalar_one_or_none()


# ── 색인 작업 ─────────────────────────────────────────────────────────────


async def claim_index_job(db: AsyncSession, document_id: int, stale_after_seconds: int) -> str | None:
    """색인 작업을 원자적으로 선점하고 새 job_id를 반환한다. 다른 작업이 진행 중이면 None.

    UPDATE 한 문장이라 동시에 들어온 첫 질문 중 하나만 성공한다 → 임베딩 호출이 중복되지 않는다.
    오래된 'indexing'은 작업이 죽은 것으로 보고 다시 선점하지만, 그 작업이 아직 살아 있어도 안전하다:
    job_id가 달라져서 옛 작업은 완료를 기록할 수 없고, 옛 작업의 청크는 검색 대상이 아니다.
    """
    job_id = uuid.uuid4().hex
    stale_before = func.now() - timedelta(seconds=stale_after_seconds)
    claimed = await db.execute(
        update(PaperDocument)
        .where(
            PaperDocument.id == document_id,
            PaperDocument.purge_pending_at.is_(None),
            or_(
                PaperDocument.index_status.in_(("none", "failed")),
                and_(PaperDocument.index_status == "indexing", PaperDocument.index_started_at < stale_before),
            ),
        )
        .values(index_status="indexing", index_job_id=job_id, index_started_at=func.now(), index_error=None)
        .returning(PaperDocument.id)
    )
    return job_id if claimed.scalar_one_or_none() is not None else None


async def finish_index_job(db: AsyncSession, document_id: int, job_id: str, chunk_count: int) -> bool:
    """색인 완료를 기록한다. 이 작업이 여전히 현재 작업이고 문서가 삭제 중이 아닐 때만 성공한다."""
    finished = await db.execute(
        update(PaperDocument)
        .where(
            PaperDocument.id == document_id,
            PaperDocument.index_job_id == job_id,
            PaperDocument.purge_pending_at.is_(None),
        )
        .values(index_status="ready", indexed_chunk_count=chunk_count)
        .returning(PaperDocument.id)
    )
    return finished.scalar_one_or_none() is not None


async def fail_index_job(db: AsyncSession, document_id: int, job_id: str, error: str) -> bool:
    """색인 실패를 기록한다. 현재 작업만 기록할 수 있다 — 오래된 작업의 실패가 새 색인을 failed로 만들지 않는다."""
    failed = await db.execute(
        update(PaperDocument)
        .where(PaperDocument.id == document_id, PaperDocument.index_job_id == job_id)
        .values(index_status="failed", index_error=error[:1000])
        .returning(PaperDocument.id)
    )
    return failed.scalar_one_or_none() is not None


async def reset_lost_index(db: AsyncSession, document_id: int, job_id: str) -> bool:
    """ready인데 벡터 색인에 청크가 없을 때(색인 유실) 다시 색인하도록 되돌린다. 확인한 그 작업일 때만 되돌린다."""
    reset = await db.execute(
        update(PaperDocument)
        .where(
            PaperDocument.id == document_id,
            PaperDocument.index_job_id == job_id,
            PaperDocument.index_status == "ready",
        )
        .values(index_status="none", index_job_id=None, indexed_chunk_count=None)
        .returning(PaperDocument.id)
    )
    return reset.scalar_one_or_none() is not None
