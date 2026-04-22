import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.user import User

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"
_EXPIRE_DAYS = 7


def hash_password(plain: str) -> str:
    """bcrypt로 비밀번호를 해싱한다."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """비밀번호를 검증한다."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: int, email: str) -> str:
    """JWT 액세스 토큰을 발급한다 (7일 만료)."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """JWT를 검증하고 payload를 반환한다. 유효하지 않으면 401 반환."""
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않은 토큰입니다.")


async def register_user(db: AsyncSession, email: str, password: str, username: str, full_name: str | None = None) -> User:
    """신규 유저를 생성한다. 이메일/유저명 중복 시 400 반환."""
    existing = await db.execute(
        select(User).where((User.email == email) | (User.username == username))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미 사용 중인 이메일 또는 유저명입니다.")

    user = User(email=email, password_hash=hash_password(password), username=username, full_name=full_name)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    """이메일과 비밀번호로 유저를 인증한다. 실패 시 401 반환."""
    result = await db.execute(select(User).where(User.email == email, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="이메일 또는 비밀번호가 올바르지 않습니다.")

    user.last_login_at = datetime.now(timezone.utc)
    return user
