from sqlalchemy import select
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
    )
    db.add(result)
    await db.flush()
    return result


async def get_recent_analysis_results(
    db: AsyncSession, limit: int = 20
) -> list[AnalysisResult]:
    """최근 분석 결과 조회"""
    result = await db.execute(
        select(AnalysisResult).order_by(AnalysisResult.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())
