import json
import logging

import httpx
from fastapi import HTTPException

from agents.log_stream import emit_log
from agents.state import AgentState
from services import arxiv_service, openalex_service, semantic_scholar_service

logger = logging.getLogger(__name__)

_HF_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def _parse_plan(plan_str: str) -> dict:
    """plan JSON 문자열을 파싱한다. 실패 시 빈 dict 반환."""
    try:
        return json.loads(plan_str)
    except (json.JSONDecodeError, TypeError):
        return {}


async def _fetch_hf_trending(limit: int = 5) -> list[dict]:
    """HuggingFace Daily Papers API에서 오늘의 트렌드 논문을 수집한다."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(_HF_PAPERS_URL, params={"limit": limit})
            response.raise_for_status()
            items = response.json()
        except Exception as e:
            logger.warning(f"HuggingFace Papers API 호출 실패: {e}")
            return []

    papers = []
    for item in items:
        paper = item.get("paper", {})
        arxiv_id = paper.get("id", "")
        papers.append({
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "authors": [a.get("name", "") for a in paper.get("authors", [])],
            "abstract": paper.get("summary", ""),
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
            "published_at": paper.get("publishedAt", ""),
            "categories": [],
            "upvotes": item.get("upvotes", 0),
            "citation_count": None,
            "tldr": None,
        })
    return papers


async def researcher_node(state: AgentState) -> dict:
    """plan을 바탕으로 논문을 수집하고 papers를 state에 저장한다."""
    mode = state["mode"]
    plan = _parse_plan(state.get("plan", "{}"))
    keywords = plan.get("search_keywords") or [state.get("user_query", "")]
    search_query = " ".join(keywords[:3])

    logger.info(f"[Researcher] 시작 — 검색어: \"{search_query}\"")

    try:
        if mode == "trend":
            emit_log("researcher", "HuggingFace Daily Papers 수집 중...")
            hf_papers = await _fetch_hf_trending(limit=5)
            emit_log("researcher", f"HuggingFace 논문 {len(hf_papers)}편 수집 완료")
            logger.info(f"[Researcher] HF 논문 {len(hf_papers)}편 수집")

            emit_log("researcher", f"arXiv 트렌드 검색 중... ({search_query})")
            arxiv_results = await arxiv_service.search_papers(search_query, max_results=5)
            arxiv_papers = [p.model_dump(mode="json") for p in arxiv_results]
            emit_log("researcher", f"arXiv 논문 {len(arxiv_papers)}편 수집 완료")
            logger.info(f"[Researcher] arXiv 논문 {len(arxiv_papers)}편 수집")

            papers = hf_papers + arxiv_papers
            emit_log("researcher", f"총 {len(papers)}편 수집 완료")
            logger.info(f"[Researcher] 완료 — 총 {len(papers)}편")

            return {
                "papers": papers,
                "final_result": {"papers": papers, "mode": "trend"},
                "current_node": "researcher",
                "error": None,
            }

        else:
            # 1순위: Semantic Scholar — rate limit 관대, TL;DR/인용수 포함
            emit_log("researcher", "Semantic Scholar 검색 중...")
            papers = await semantic_scholar_service.search_papers(search_query, max_results=5)
            logger.info(f"[Researcher] S2 {len(papers)}편 수집")

            if papers:
                emit_log("researcher", f"Semantic Scholar {len(papers)}편 수집 완료")
            else:
                # 2순위: OpenAlex — S2 결과 없을 때 사용, 완전 무료
                emit_log("researcher", "Semantic Scholar 결과 없음 — OpenAlex 검색 중...")
                papers = await openalex_service.search_papers(search_query, max_results=5)
                logger.info(f"[Researcher] OpenAlex {len(papers)}편 수집")

                if papers:
                    emit_log("researcher", f"OpenAlex {len(papers)}편 수집 완료")
                else:
                    # 3순위: arXiv — 캐시/백오프 적용된 최후 fallback
                    emit_log("researcher", "OpenAlex 결과 없음 — arXiv 검색 중...")
                    arxiv_results = await arxiv_service.search_papers(search_query, max_results=5)
                    logger.info(f"[Researcher] arXiv {len(arxiv_results)}편 수집 완료")
                    emit_log("researcher", f"arXiv {len(arxiv_results)}편 수집 완료")

                    emit_log("researcher", "인용 정보 보완 중...")
                    arxiv_ids = [p.arxiv_id for p in arxiv_results]
                    enrichments = await semantic_scholar_service.enrich_papers(arxiv_ids)
                    logger.info(f"[Researcher] S2 보완 완료 — {len(enrichments)}/{len(arxiv_ids)}편 매칭")

                    papers = []
                    for paper in arxiv_results:
                        paper_dict = paper.model_dump(mode="json")
                        base_id = paper.arxiv_id.split("v")[0]
                        if s2 := enrichments.get(base_id):
                            paper_dict["citation_count"] = s2.citation_count
                            paper_dict["influential_citation_count"] = s2.influential_citation_count
                            paper_dict["tldr"] = s2.tldr
                            paper_dict["s2_paper_id"] = s2.s2_paper_id
                        papers.append(paper_dict)

            titles = [p.get("title", "")[:50] for p in papers]
            for i, t in enumerate(titles, 1):
                logger.info(f"[Researcher]   {i}. {t}")
                emit_log("researcher", f"{i}. {t}")

    except HTTPException as e:
        logger.error(f"[Researcher] 논문 수집 실패: {e.detail}")
        return {
            "papers": [],
            "current_node": "researcher",
            "error": f"논문 수집 실패: {e.detail}",
        }
    except Exception as e:
        logger.error(f"[Researcher] 예기치 않은 오류: {e}")
        return {
            "papers": [],
            "current_node": "researcher",
            "error": f"논문 수집 중 오류 발생: {str(e)}",
        }

    logger.info(f"[Researcher] 완료 — {len(papers)}편")
    return {
        "papers": papers,
        "current_node": "researcher",
        "error": None,
    }
