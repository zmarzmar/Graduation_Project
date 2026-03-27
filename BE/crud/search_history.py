from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.search_history import SearchHistory


async def create_search_history(
    db: AsyncSession,
    query: str,
    mode: str,
    result_count: int = 0,
) -> SearchHistory:
    """검색 기록 저장"""
    history = SearchHistory(query=query, mode=mode, result_count=result_count)
    db.add(history)
    await db.flush()
    return history


async def get_recent_search_history(
    db: AsyncSession, limit: int = 20
) -> list[SearchHistory]:
    """최근 검색 기록 조회"""
    result = await db.execute(
        select(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())
