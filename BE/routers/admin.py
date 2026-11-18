from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_admin_user, get_db
from models.user import User
from services import admin_service, ai_switch_service

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminUserItem(BaseModel):
    id: int
    email: str
    username: str
    full_name: str | None
    affiliation: str | None
    preferred_framework: list[str] | None
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


class AiSwitchStatus(BaseModel):
    enabled: bool
    updated_at: datetime | None  # 한 번도 바꾸지 않았으면 None
    today_analyses: int  # 오늘(한국 시간) 저장된 분석 기록 — 저장되지 않은 요청(저장 실패 등)은 빠진다
    today_searches: int  # 오늘 저장된 검색·트렌드 기록


class AiSwitchUpdate(BaseModel):
    enabled: bool


@router.get("/ai-switch", response_model=AiSwitchStatus)
async def get_ai_switch(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
):
    """AI 사용 중지 스위치 상태와 오늘 사용량."""
    return await ai_switch_service.get_status(db)


@router.put("/ai-switch", response_model=AiSwitchStatus)
async def put_ai_switch(
    body: AiSwitchUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
):
    """AI 기능을 켜거나 끈다. 끄면 검색·분석·PDF·트렌드·Q&A 요청이 503이 된다 (진행 중인 분석은 끝까지 돈다)."""
    return await ai_switch_service.set_ai_enabled(db, body.enabled)
