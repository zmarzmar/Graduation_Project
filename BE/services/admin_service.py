from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.analysis import AnalysisResult
from models.paper import Paper
from models.search_history import SearchHistory
from models.user import User


async def list_users(db: AsyncSession) -> list[User]:
    """관리자 화면용 사용자 목록을 최신 가입순으로 조회한다."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return list(result.scalars().all())


async def list_papers(db: AsyncSession) -> list[Paper]:
    """관리자 화면용 저장 논문 목록을 최신 저장순으로 조회한다."""
    result = await db.execute(select(Paper).order_by(Paper.created_at.desc()).limit(100))
    return list(result.scalars().all())


async def get_system_summary(db: AsyncSession) -> dict:
    """관리자 화면용 시스템/데이터 요약을 조회한다."""
    user_count = await db.scalar(select(func.count()).select_from(User))
    active_user_count = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active == True)  # noqa: E712
    )
    admin_user_count = await db.scalar(
        select(func.count()).select_from(User).where(User.is_admin == True)  # noqa: E712
    )
    paper_count = await db.scalar(select(func.count()).select_from(Paper))
    search_history_count = await db.scalar(
        select(func.count()).select_from(SearchHistory).where(SearchHistory.is_deleted == False)  # noqa: E712
    )
    analysis_count = await db.scalar(
        select(func.count()).select_from(AnalysisResult).where(AnalysisResult.is_deleted == False)  # noqa: E712
    )

    return {
        "status": "ok",
        "version": "0.1.0",
        "database": "ok",
        "counts": {
            "users": user_count or 0,
            "active_users": active_user_count or 0,
            "admin_users": admin_user_count or 0,
            "papers": paper_count or 0,
            "search_history": search_history_count or 0,
            "analysis_results": analysis_count or 0,
        },
    }
