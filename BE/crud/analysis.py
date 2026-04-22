from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

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
) -> AnalysisResult:
    """분석 결과 저장"""
    result = AnalysisResult(
        paper_id=paper_id,
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
    return result.rowcount
