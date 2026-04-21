from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
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


_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    """로그인 필수 의존성 — 토큰 없거나 유효하지 않으면 401 반환."""
    from models.user import User
    from services.auth_service import decode_access_token

    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")

    payload = decode_access_token(credentials.credentials)
    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유저를 찾을 수 없습니다.")
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    """로그인 선택 의존성 — 토큰 있으면 유저 반환, 없으면 None 반환."""
    from models.user import User
    from services.auth_service import decode_access_token

    if not credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
        result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))  # noqa: E712
        return result.scalar_one_or_none()
    except Exception:
        return None
