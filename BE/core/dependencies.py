from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings

# async 엔진 생성
engine = create_async_engine(
    settings.database_url,
    echo=False,  # SQL 로그 출력 여부 (디버깅 시 True로 변경)
    pool_pre_ping=True,  # 연결 유효성 사전 확인
)

# 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 의존성 — 요청마다 DB 세션을 생성하고 자동으로 닫는다."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
