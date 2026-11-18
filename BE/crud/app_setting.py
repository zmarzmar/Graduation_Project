from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.app_setting import AppSetting


async def get_setting(db: AsyncSession, key: str) -> AppSetting | None:
    return await db.get(AppSetting, key)


async def set_setting(db: AsyncSession, key: str, enabled: bool) -> AppSetting:
    """설정을 쓴다. 행이 없으면 만든다 (commit은 호출자가 한다)."""
    stmt = (
        insert(AppSetting)
        .values(key=key, enabled=enabled)
        .on_conflict_do_update(index_elements=[AppSetting.key], set_={"enabled": enabled, "updated_at": func.now()})
        .returning(AppSetting)
    )
    return (await db.execute(stmt)).scalar_one()
