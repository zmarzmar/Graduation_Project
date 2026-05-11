import logging

from fastapi import APIRouter, Query
from pydantic import BaseModel

from schemas.paper import SearchResponse
from services import arxiv_service, semantic_scholar_service, keyword_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["papers"])


class DailyKeywordsResponse(BaseModel):
    keywords: list[str]


@router.get("/papers/daily-keywords", response_model=DailyKeywordsResponse)
async def get_daily_keywords() -> DailyKeywordsResponse:
    """오늘 날짜 기준 AI 트렌드 키워드 5개 반환 (당일 캐시)"""
    keywords = await keyword_service.get_daily_keywords()
    return DailyKeywordsResponse(keywords=keywords)


@router.get("/papers/search", response_model=SearchResponse)
async def search_papers(
    search: str = Query(..., min_length=1, description="Search query (e.g. 'attention mechanism', 'LoRA fine-tuning')"),
    max_results: int = Query(3, ge=1, le=3, description="Number of results (max 3)"),
) -> SearchResponse:
    """arXiv 논문 검색 + Semantic Scholar 인용 정보 보완"""
    papers = await arxiv_service.search_papers(search, max_results)

    try:
        enrichments = await semantic_scholar_service.enrich_papers([p.arxiv_id for p in papers])
    except Exception as e:
        logger.warning(f"[Enrich] 인용 정보 보강 실패 (검색은 계속): {e}")
        enrichments = {}

    for paper in papers:
        base_id = paper.arxiv_id.split("v")[0]
        if data := enrichments.get(base_id):
            paper.citation_count = data.citation_count
            paper.influential_citation_count = data.influential_citation_count
            paper.tldr = data.tldr
            paper.s2_paper_id = data.s2_paper_id

    return SearchResponse(query=search, total=len(papers), papers=papers)
