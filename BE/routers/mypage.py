from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import get_db
from crud import analysis as crud_analysis
from crud import search_history as crud_search_history
from models.analysis import AnalysisResult
from models.paper import Paper
from models.search_history import SearchHistory
from models.user import User

router = APIRouter(tags=["mypage"])


class UserInfo(BaseModel):
    id: int
    username: str
    full_name: str | None
    email: str
    affiliation: str | None
    preferred_framework: str | None

    class Config:
        from_attributes = True


@router.get("/mypage/me", response_model=UserInfo)
async def get_my_info(db: AsyncSession = Depends(get_db)):
    """내 정보 조회 — 로그인 미구현으로 임시 id=1 반환"""
    result = await db.execute(select(User).where(User.id == 1))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다.")
    return user


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


class AnalysisDetail(BaseModel):
    id: int
    query: str
    mode: str
    paper_title: str | None
    paper_authors: list[str] | None
    paper_summary: str
    paper_review: dict
    key_formulas: list
    generated_code: str
    review_feedback: str
    review_passed: bool
    iteration_count: int
    created_at: datetime


@router.get("/mypage/analysis-history/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis_detail(analysis_id: int, db: AsyncSession = Depends(get_db)):
    """특정 분석 결과 상세 조회"""
    result = await db.execute(
        select(AnalysisResult, Paper)
        .outerjoin(Paper, AnalysisResult.paper_id == Paper.id)
        .where(AnalysisResult.id == analysis_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")

    analysis, paper = row
    return AnalysisDetail(
        id=analysis.id,
        query=analysis.query,
        mode=analysis.mode,
        paper_title=paper.title if paper else None,
        paper_authors=paper.authors if paper else None,
        paper_summary=analysis.paper_summary,
        paper_review=analysis.paper_review,
        key_formulas=analysis.key_formulas,
        generated_code=analysis.generated_code,
        review_feedback=analysis.review_feedback,
        review_passed=analysis.review_passed,
        iteration_count=analysis.iteration_count,
        created_at=analysis.created_at,
    )


@router.get("/mypage/search-history", response_model=list[SearchHistoryItem])
async def get_search_history(db: AsyncSession = Depends(get_db)):
    """최근 검색 기록 20개 반환"""
    result = await db.execute(
        select(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(20)
    )
    return result.scalars().all()


@router.delete("/mypage/search-history", status_code=204)
async def delete_all_search_history(db: AsyncSession = Depends(get_db)):
    """검색 기록 전체 삭제"""
    await crud_search_history.delete_all_search_history(db)
    await db.commit()


@router.delete("/mypage/search-history/{history_id}", status_code=204)
async def delete_search_history(history_id: int, db: AsyncSession = Depends(get_db)):
    """검색 기록 개별 삭제"""
    deleted = await crud_search_history.delete_search_history_by_id(db, history_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="검색 기록을 찾을 수 없습니다.")
    await db.commit()


@router.delete("/mypage/analysis-history", status_code=204)
async def delete_all_analysis_history(db: AsyncSession = Depends(get_db)):
    """분석 히스토리 전체 삭제"""
    await crud_analysis.delete_all_analysis_results(db)
    await db.commit()


@router.delete("/mypage/analysis-history/{analysis_id}", status_code=204)
async def delete_analysis_history(analysis_id: int, db: AsyncSession = Depends(get_db)):
    """분석 히스토리 개별 삭제"""
    deleted = await crud_analysis.delete_analysis_result_by_id(db, analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    await db.commit()


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
