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


async def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> int | None:
    """로그인 선택 의존성 — 토큰이 있으면 검증 후 user_id 반환, 없으면 None.

    - 헤더 없음: None
    - 토큰 불량/만료: decode_access_token이 던지는 HTTPException(401)을 그대로 전파
    - sub 클레임 손상: 401
    - 유저 없음/비활성: 401

    SSE 엔드포인트에서 사용하므로 DB 세션을 스트림 수명과 묶지 않기 위해
    AsyncSessionLocal()을 함수 내부에서 짧게 열어 조회 후 즉시 닫는다.
    """
    from models.user import User
    from services.auth_service import decode_access_token

    if not credentials:
        return None

    payload = decode_access_token(credentials.credentials)
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
        )
        user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유저를 찾을 수 없습니다.",
        )
    return user_id
