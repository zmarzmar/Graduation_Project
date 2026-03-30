from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import get_db
from models.analysis import AnalysisResult
from models.paper import Paper
from models.search_history import SearchHistory

router = APIRouter(tags=["mypage"])


class SearchHistoryItem(BaseModel):
    id: int
    query: str
    mode: str
    result_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisHistoryItem(BaseModel):
    id: int
    query: str
    mode: str
    paper_title: str | None
    paper_authors: list[str] | None
    review_passed: bool
    has_code: bool
    created_at: datetime


@router.get("/mypage/search-history", response_model=list[SearchHistoryItem])
async def get_search_history(db: AsyncSession = Depends(get_db)):
    """최근 검색 기록 20개 반환"""
    result = await db.execute(
        select(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(20)
    )
    return result.scalars().all()


@router.get("/mypage/analysis-history", response_model=list[AnalysisHistoryItem])
async def get_analysis_history(db: AsyncSession = Depends(get_db)):
    """최근 분석 히스토리 20개 반환 (논문 제목 포함)"""
    result = await db.execute(
        select(AnalysisResult, Paper)
        .outerjoin(Paper, AnalysisResult.paper_id == Paper.id)
        .order_by(AnalysisResult.created_at.desc())
        .limit(20)
    )
    rows = result.all()

    items = []
    for analysis, paper in rows:
        items.append(AnalysisHistoryItem(
            id=analysis.id,
            query=analysis.query,
            mode=analysis.mode,
            paper_title=paper.title if paper else None,
            paper_authors=paper.authors if paper else None,
            review_passed=analysis.review_passed,
            has_code=bool(analysis.generated_code),
            created_at=analysis.created_at,
        ))
    return items
