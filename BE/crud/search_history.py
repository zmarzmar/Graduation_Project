from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.search_history import SearchHistory


async def create_search_history(
    db: AsyncSession,
    query: str,
    mode: str,
    result_count: int = 0,
    papers: list[dict] | None = None,
) -> SearchHistory:
    """검색 기록 저장"""
    # 논문 목록에서 표시에 필요한 최소 정보만 저장
    paper_list = [
        {
            "title": p.get("title", ""),
            "authors": p.get("authors", [])[:3],
            "arxiv_id": p.get("arxiv_id", ""),
            "url": p.get("url", ""),
        }
        for p in (papers or [])
    ]
    history = SearchHistory(query=query, mode=mode, result_count=result_count, papers=paper_list)
    db.add(history)
    await db.flush()
    return history


async def get_recent_search_history(
    db: AsyncSession, limit: int = 20
) -> list[SearchHistory]:
    """최근 검색 기록 조회 (소프트 딜리트 제외)"""
    result = await db.execute(
        select(SearchHistory)
        .where(SearchHistory.is_deleted == False)  # noqa: E712
        .order_by(SearchHistory.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def delete_search_history_by_id(db: AsyncSession, history_id: int) -> bool:
    """검색 기록 개별 소프트 딜리트. 성공 여부 반환"""
    result = await db.execute(
        select(SearchHistory).where(SearchHistory.id == history_id, SearchHistory.is_deleted == False)  # noqa: E712
    )
    obj = result.scalar_one_or_none()
    if not obj:
        return False
    obj.is_deleted = True
    return True


async def delete_all_search_history(db: AsyncSession) -> int:
    """검색 기록 전체 소프트 딜리트. 처리된 행 수 반환"""
    result = await db.execute(
        update(SearchHistory)
        .where(SearchHistory.is_deleted == False)  # noqa: E712
        .values(is_deleted=True)
    )
    return result.rowcount
