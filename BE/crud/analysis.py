from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from crud.paper_document import mark_unreferenced_documents_for_purge
from crud.qa_message import delete_for_analysis, delete_for_user
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
    summary_claims: list | None = None,
    paper_review: dict | None = None,
    key_formulas: list | None = None,
    user_id: int | None = None,
    document_id: int | None = None,
    code_skip_reason: str = "",
    exec_status: str = "",
    exec_detail: str = "",
    run_error: str = "",
) -> AnalysisResult:
    """분석 결과 저장"""
    result = AnalysisResult(
        paper_id=paper_id,
        document_id=document_id,
        mode=mode,
        query=query,
        generated_code=generated_code,
        review_feedback=review_feedback,
        code_skip_reason=code_skip_reason,
        exec_status=exec_status,
        exec_detail=exec_detail,
        run_error=run_error,
        review_passed=review_passed,
        iteration_count=iteration_count,
        paper_summary=paper_summary,
        summary_claims=summary_claims or [],
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


async def get_key_formulas(db: AsyncSession, result_id: int, user_id: int) -> list:
    """본인 소유의 삭제되지 않은 분석 기록의 핵심 수식 (Analyzer가 뽑은 것). 없으면 빈 목록."""
    found = await db.scalar(
        select(AnalysisResult.key_formulas).where(
            AnalysisResult.id == result_id,
            AnalysisResult.user_id == user_id,
            AnalysisResult.is_deleted == False,  # noqa: E712
        )
    )
    return found or []


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
    # 그 분석의 Q&A도 지운다 — 분석 행 UPDATE가 먼저라, 답을 기다리던 저장은 이 커밋 뒤 '삭제됨'을 보고 저장하지 않는다
    await delete_for_analysis(db, result_id)
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
    await delete_for_user(db, user_id)  # 모든 분석이 지워졌으니 Q&A도 모두
    # 활성 기록이 없어졌으므로 사용자의 원문을 모두 비우고 삭제 대기로 표시한다
    await mark_unreferenced_documents_for_purge(db, user_id)
    return result.rowcount


async def count_created_since(db: AsyncSession, since: datetime) -> int:
    """since 이후에 만든 분석 기록 수 — 삭제한 기록도 센다 (이미 비용이 든 요청)."""
    return await db.scalar(select(func.count()).select_from(AnalysisResult).where(AnalysisResult.created_at >= since)) or 0
