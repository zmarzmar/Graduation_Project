from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_admin_user, get_db
from models.user import User
from services import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminUserItem(BaseModel):
    id: int
    email: str
    username: str
    full_name: str | None
    affiliation: str | None
    preferred_framework: str | None
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login_at: datetime | None

    class Config:
        from_attributes = True


class AdminPaperItem(BaseModel):
    id: int
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    pdf_url: str
    published_at: datetime | None
    categories: list[str]
    citation_count: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class AdminCounts(BaseModel):
    users: int
    active_users: int
    admin_users: int
    papers: int
    search_history: int
    analysis_results: int


class AdminSystemSummary(BaseModel):
    status: str
    version: str
    database: str
    counts: AdminCounts


@router.get("/users", response_model=list[AdminUserItem])
async def get_admin_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
):
    """관리자 사용자 목록 조회."""
    return await admin_service.list_users(db)


@router.get("/papers", response_model=list[AdminPaperItem])
async def get_admin_papers(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
):
    """관리자 저장 논문 목록 조회."""
    return await admin_service.list_papers(db)


@router.get("/system", response_model=AdminSystemSummary)
async def get_admin_system(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
):
    """관리자 시스템/데이터 요약 조회."""
    return await admin_service.get_system_summary(db)
