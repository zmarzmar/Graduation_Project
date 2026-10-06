from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from crud.paper_document import mark_unreferenced_documents_for_purge
from models.analysis import AnalysisResult


async def create_analysis_result(
    db: AsyncSession,
    mode: str,
    query: str,
    generated_code: str,
    review_feedback: str,
    review_passed: bool,
    iteration_count: int,
    paper_id: int | None = None,
    paper_summary: str = "",
    paper_review: dict | None = None,
    key_formulas: list | None = None,
    user_id: int | None = None,
    document_id: int | None = None,
) -> AnalysisResult:
    """분석 결과 저장"""
    result = AnalysisResult(
        paper_id=paper_id,
        document_id=document_id,
        mode=mode,
        query=query,
        generated_code=generated_code,
        review_feedback=review_feedback,
        review_passed=review_passed,
        iteration_count=iteration_count,
        paper_summary=paper_summary,
        paper_review=paper_review or {},
        key_formulas=key_formulas or [],
        user_id=user_id,
    )
    db.add(result)
    await db.flush()
    return result


async def get_recent_analysis_results(
    db: AsyncSession, limit: int = 20
) -> list[AnalysisResult]:
    """최근 분석 결과 조회 (소프트 딜리트 제외)"""
    result = await db.execute(
        select(AnalysisResult)
        .where(AnalysisResult.is_deleted == False)  # noqa: E712
        .order_by(AnalysisResult.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def user_has_active_analysis(db: AsyncSession, result_id: int, user_id: int) -> bool:
    """본인 소유의 삭제되지 않은 분석 기록인지 확인한다."""
    found = await db.scalar(
        select(AnalysisResult.id).where(
            AnalysisResult.id == result_id,
            AnalysisResult.user_id == user_id,
            AnalysisResult.is_deleted == False,  # noqa: E712
        )
    )
    return found is not None


async def delete_analysis_result_by_id(db: AsyncSession, result_id: int, user_id: int) -> bool:
    """분석 결과 개별 소프트 딜리트. 본인 소유만 가능. 성공 여부 반환"""
    result = await db.execute(
        select(AnalysisResult).where(
            AnalysisResult.id == result_id,
            AnalysisResult.user_id == user_id,
            AnalysisResult.is_deleted == False,  # noqa: E712
        )
    )
    obj = result.scalar_one_or_none()
    if not obj:
        return False
    obj.is_deleted = True
    # 소프트 삭제를 먼저 반영해야 아래 정리 쿼리가 이 기록을 '삭제됨'으로 본다
    await db.flush()
    # 이 기록이 문서를 가리키던 마지막 활성 기록이었다면 원문을 비우고 삭제 대기로 표시한다.
    # 벡터 색인 삭제와 행 제거는 커밋 뒤에 rag_service.purge_pending_documents가 한다.
    await mark_unreferenced_documents_for_purge(db, user_id)
    return True


async def delete_all_analysis_results(db: AsyncSession, user_id: int) -> int:
    """특정 유저의 분석 결과 전체 소프트 딜리트. 처리된 행 수 반환"""
    result = await db.execute(
        update(AnalysisResult)
        .where(
            AnalysisResult.user_id == user_id,
            AnalysisResult.is_deleted == False,  # noqa: E712
        )
        .values(is_deleted=True)
    )
    # 활성 기록이 없어졌으므로 사용자의 원문을 모두 비우고 삭제 대기로 표시한다
    await mark_unreferenced_documents_for_purge(db, user_id)
    return result.rowcount
