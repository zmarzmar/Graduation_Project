from sqlalchemy import delete, select
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


async def delete_search_history_by_id(db: AsyncSession, history_id: int) -> bool:
    """검색 기록 개별 삭제. 삭제 성공 여부 반환"""
    result = await db.execute(
        select(SearchHistory).where(SearchHistory.id == history_id)
    )
    obj = result.scalar_one_or_none()
    if not obj:
        return False
    await db.delete(obj)
    return True


async def delete_all_search_history(db: AsyncSession) -> int:
    """검색 기록 전체 삭제. 삭제된 행 수 반환"""
    result = await db.execute(delete(SearchHistory))
    return result.rowcount
