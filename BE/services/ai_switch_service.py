"""AI 사용 중지 스위치 — 관리자가 끄면 비용이 드는 요청(검색·분석·PDF·트렌드·Q&A)을 503으로 막는다."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from crud import app_setting as crud_app_setting
from models.analysis import AnalysisResult
from models.search_history import SearchHistory

AI_ENABLED_KEY = "ai_enabled"
AI_DISABLED_MESSAGE = "지금은 이용자가 많아 AI 분석을 잠시 멈췄어요. 잠시 후 다시 시도해 주세요."
# '오늘'은 한국 시간 자정부터 센다 (전시·사용자 모두 한국)
_KST = timezone(timedelta(hours=9))


async def is_ai_enabled(db: AsyncSession) -> bool:
    """행이 없으면 켜진 것으로 본다 — 스위치가 없던 상태와 같게."""
    setting = await crud_app_setting.get_setting(db, AI_ENABLED_KEY)
    return True if setting is None else setting.enabled


async def get_status(db: AsyncSession) -> dict:
    """스위치 상태와 오늘(한국 시간) 저장된 분석·검색 기록 수."""
    setting = await crud_app_setting.get_setting(db, AI_ENABLED_KEY)
    start = datetime.now(_KST).replace(hour=0, minute=0, second=0, microsecond=0)
    # 삭제한 기록도 이미 비용이 든 요청이라 센다
    analyses = await db.scalar(select(func.count()).select_from(AnalysisResult).where(AnalysisResult.created_at >= start))
    searches = await db.scalar(select(func.count()).select_from(SearchHistory).where(SearchHistory.created_at >= start))
    return {
        "enabled": True if setting is None else setting.enabled,
        "updated_at": setting.updated_at if setting else None,
        "today_analyses": analyses or 0,
        "today_searches": searches or 0,
    }


async def set_ai_enabled(db: AsyncSession, enabled: bool) -> dict:
    await crud_app_setting.set_setting(db, AI_ENABLED_KEY, enabled)
    await db.commit()
    return await get_status(db)
