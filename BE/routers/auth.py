from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_user, get_db
from models.user import User
from services import auth_service

router = APIRouter(tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    username: str = Field(..., min_length=2, max_length=50)
    full_name: str | None = Field(None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str | None
    affiliation: str | None
    preferred_framework: str | None

    class Config:
        from_attributes = True


@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """회원가입 — 성공 시 JWT 즉시 반환."""
    user = await auth_service.register_user(db, request.email, request.password, request.username, request.full_name)
    await db.commit()
    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """로그인 — 성공 시 JWT 반환."""
    user = await auth_service.authenticate_user(db, request.email, request.password)
    await db.commit()
    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """현재 로그인된 유저 정보 반환."""
    return current_user
