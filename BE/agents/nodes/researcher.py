import asyncio
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

            # S2 보완 (5편) — 실패해도 HF 논문 유지
            s2_papers = []
            try:
                emit_log("researcher", f"Semantic Scholar 트렌드 검색 중... ({search_query})")
                s2_papers = await semantic_scholar_service.search_papers(search_query, max_results=5)
                emit_log("researcher", f"Semantic Scholar {len(s2_papers)}편 수집 완료")
                logger.info(f"[Researcher] S2 트렌드 {len(s2_papers)}편 수집")
            except Exception as e:
                logger.warning(f"[Researcher] S2 수집 실패 (무시): {e}")
                emit_log("researcher", "Semantic Scholar 수집 실패 — 계속 진행")

            # arXiv 보너스 (5편) — 성공하면 최대 15편, 실패해도 10편 보장
            arxiv_papers = []
            try:
                emit_log("researcher", f"arXiv 트렌드 검색 중... ({search_query})")
                arxiv_results = await arxiv_service.search_papers(search_query, max_results=5)
                arxiv_papers = [p.model_dump(mode="json") for p in arxiv_results]
                emit_log("researcher", f"arXiv {len(arxiv_papers)}편 수집 완료")
                logger.info(f"[Researcher] arXiv 트렌드 {len(arxiv_papers)}편 수집")
            except Exception as e:
                logger.warning(f"[Researcher] arXiv 수집 실패 (무시): {e}")
                emit_log("researcher", "arXiv 수집 실패 — HF+S2 논문으로 계속 진행")

            papers = hf_papers + s2_papers + arxiv_papers
            emit_log("researcher", f"총 {len(papers)}편 수집 완료")
            logger.info(f"[Researcher] 완료 — 총 {len(papers)}편")

            return {
                "papers": papers,
                "final_result": {"papers": papers, "mode": "trend"},
                "current_node": "researcher",
                "error": None,
            }

        else:
            # S2 + OpenAlex 병렬 수집 (각 최대 5개) → 합쳐서 최대 10개
            emit_log("researcher", "Semantic Scholar + OpenAlex 동시 검색 중...")
            s2_task = asyncio.create_task(
                semantic_scholar_service.search_papers(search_query, max_results=5)
            )
            oa_task = asyncio.create_task(
                openalex_service.search_papers(search_query, max_results=5)
            )
            s2_results, oa_results = await asyncio.gather(s2_task, oa_task, return_exceptions=True)

            # 예외 처리 — 실패한 소스는 빈 리스트로 처리
            if isinstance(s2_results, Exception):
                logger.warning(f"[Researcher] S2 수집 실패: {s2_results}")
                s2_results = []
            if isinstance(oa_results, Exception):
                logger.warning(f"[Researcher] OpenAlex 수집 실패: {oa_results}")
                oa_results = []

            logger.info(f"[Researcher] S2 {len(s2_results)}편, OpenAlex {len(oa_results)}편 수집")
            emit_log("researcher", f"Semantic Scholar {len(s2_results)}편, OpenAlex {len(oa_results)}편 수집 완료")

            # 중복 제거 — arxiv_id 우선, 없으면 title 기준 / S2 결과 우선 유지
            seen_ids: set[str] = set()
            seen_titles: set[str] = set()
            papers: list[dict] = []

            def _add_if_unique(p: dict) -> None:
                aid = (p.get("arxiv_id") or "").split("v")[0]
                title = (p.get("title") or "").strip().lower()
                if aid:
                    if aid in seen_ids:
                        return
                    seen_ids.add(aid)
                elif title:
                    if title in seen_titles:
                        return
                    seen_titles.add(title)
                papers.append(p)

            for p in s2_results:
                _add_if_unique(p)
            for p in oa_results:
                _add_if_unique(p)

            # 10개 미만이면 arXiv로 부족분 채움
            shortage = 10 - len(papers)
            if shortage > 0:
                emit_log("researcher", f"논문 {len(papers)}편 — arXiv에서 {shortage}편 추가 수집 중...")
                try:
                    arxiv_results = await arxiv_service.search_papers(search_query, max_results=shortage)
                    logger.info(f"[Researcher] arXiv {len(arxiv_results)}편 수집")

                    # arXiv 논문에 S2 인용 정보 보완
                    arxiv_ids = [p.arxiv_id for p in arxiv_results]
                    enrichments = await semantic_scholar_service.enrich_papers(arxiv_ids)
                    logger.info(f"[Researcher] S2 보완 완료 — {len(enrichments)}/{len(arxiv_ids)}편 매칭")

                    for paper in arxiv_results:
                        aid = paper.arxiv_id.split("v")[0]
                        if aid in seen_ids:
                            continue
                        paper_dict = paper.model_dump(mode="json")
                        if s2 := enrichments.get(aid):
                            paper_dict["citation_count"] = s2.citation_count
                            paper_dict["influential_citation_count"] = s2.influential_citation_count
                            paper_dict["tldr"] = s2.tldr
                            paper_dict["s2_paper_id"] = s2.s2_paper_id
                        papers.append(paper_dict)
                        seen_ids.add(aid)

                    emit_log("researcher", f"arXiv {len(arxiv_results)}편 추가 완료")
                except Exception as e:
                    logger.warning(f"[Researcher] arXiv 수집 실패 (무시): {e}")
                    emit_log("researcher", "arXiv 수집 실패 — 계속 진행")

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
