from fastapi import APIRouter, Query

from schemas.paper import SearchResponse
from services import arxiv_service, semantic_scholar_service

router = APIRouter(tags=["papers"])


@router.get("/papers/search", response_model=SearchResponse)
async def search_papers(
    search: str = Query(..., min_length=1, description="Search query (e.g. 'attention mechanism', 'LoRA fine-tuning')"),
    max_results: int = Query(3, ge=1, le=3, description="Number of results (max 3)"),
) -> SearchResponse:
    """arXiv 논문 검색 + Semantic Scholar 인용 정보 보완"""
    papers = await arxiv_service.search_papers(search, max_results)

    enrichments = await semantic_scholar_service.enrich_papers([p.arxiv_id for p in papers])

    for paper in papers:
        base_id = paper.arxiv_id.split("v")[0]
        if data := enrichments.get(base_id):
            paper.citation_count = data.citation_count
            paper.influential_citation_count = data.influential_citation_count
            paper.tldr = data.tldr
            paper.s2_paper_id = data.s2_paper_id

    return SearchResponse(query=search, total=len(papers), papers=papers)
