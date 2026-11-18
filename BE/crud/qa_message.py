from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.analysis import AnalysisResult
from models.qa_message import QaMessage

RECENT_LIMIT = 50


async def save_for_active_analysis(db: AsyncSession, analysis_id: int, user_id: int, question: str, response: dict) -> bool:
    """본인의 삭제되지 않은 분석 기록에만 저장한다 (commit은 호출자가 한다).

    분석 행을 FOR UPDATE로 잠근다 — 답을 기다리는 동안 다른 탭에서 분석을 지웠다면 삭제가 커밋된 뒤 is_deleted를 보고
    저장하지 않는다. 삭제가 나중이면 삭제가 이 행까지 지운다 (crud/analysis의 삭제가 같은 분석 행을 UPDATE한다).
    """
    found = await db.scalar(
        select(AnalysisResult.id)
        .where(AnalysisResult.id == analysis_id, AnalysisResult.user_id == user_id,
               AnalysisResult.is_deleted == False)  # noqa: E712
        .with_for_update()
    )
    if found is None:
        return False
    db.add(QaMessage(analysis_id=analysis_id, user_id=user_id, question=question, response=response))
    await db.flush()
    return True


async def list_recent(db: AsyncSession, analysis_id: int, user_id: int) -> list[QaMessage]:
    """그 분석의 최근 Q&A (오래된 순, 최대 50개). 소유·삭제 여부는 호출자가 먼저 확인한다."""
    rows = await db.execute(
        select(QaMessage)
        .where(QaMessage.analysis_id == analysis_id, QaMessage.user_id == user_id)
        .order_by(QaMessage.created_at.desc(), QaMessage.id.desc())
        .limit(RECENT_LIMIT)
    )
    return list(reversed(rows.scalars().all()))


async def delete_for_analysis(db: AsyncSession, analysis_id: int) -> None:
    await db.execute(delete(QaMessage).where(QaMessage.analysis_id == analysis_id))


async def delete_for_analyses(db: AsyncSession, analysis_ids: list[int]) -> None:
    if analysis_ids:
        await db.execute(delete(QaMessage).where(QaMessage.analysis_id.in_(analysis_ids)))
