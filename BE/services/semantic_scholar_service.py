import logging
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from core.config import settings

logger = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_BATCH_URL  = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_SEARCH_FIELDS = (
    "title,abstract,authors,year,externalIds,"
    "citationCount,influentialCitationCount,tldr,openAccessPdf"
)
_S2_BATCH_FIELDS = "citationCount,influentialCitationCount,tldr,externalIds"


def _build_headers() -> dict[str, str]:
    """API 키가 있으면 헤더에 포함한다."""
    headers: dict[str, str] = {
        "User-Agent": "arxiv-analyst/0.1 (graduation-project; contact@example.com)"
    }
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    return headers


def _s2_item_to_dict(item: dict) -> dict:
    """S2 검색 결과 1건을 papers state 형식의 dict로 변환한다."""
    arxiv_id = (item.get("externalIds") or {}).get("ArXiv", "")
    year = item.get("year") or 2024
    pdf_url = ""
    if oa := item.get("openAccessPdf"):
        pdf_url = oa.get("url", "")
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return {
        "arxiv_id": arxiv_id,
        "title": item.get("title") or "",
        "authors": [a.get("name", "") for a in (item.get("authors") or [])],
        "abstract": item.get("abstract") or "",
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "pdf_url": pdf_url,
        "published_at": f"{year}-01-01T00:00:00",
        "categories": [],
        "citation_count": item.get("citationCount"),
        "influential_citation_count": item.get("influentialCitationCount"),
        "tldr": (item.get("tldr") or {}).get("text"),
        "s2_paper_id": item.get("paperId"),
    }


async def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """Semantic Scholar 검색 API로 논문을 검색한다.

    arXiv보다 훨씬 관대한 rate limit. API 키 없이도 동작하며,
    키가 있으면 초당 100 요청으로 상향된다.
    실패 시 빈 리스트를 반환해 arXiv fallback이 동작하도록 한다.
    """
    params = {
        "query": query,
        "limit": max_results,
        "fields": _S2_SEARCH_FIELDS,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                _S2_SEARCH_URL,
                params=params,
                headers=_build_headers(),
            )
            response.raise_for_status()
        items = response.json().get("data", [])
        results = [_s2_item_to_dict(item) for item in items if item.get("title")]
        logger.info(f"[S2 Search] '{query}' — {len(results)}편")
        return results
    except httpx.HTTPStatusError as e:
        logger.warning(f"[S2 Search] HTTP {e.response.status_code} — arXiv fallback 사용")
        return []
    except Exception as e:
        logger.warning(f"[S2 Search] 실패: {e} — arXiv fallback 사용")
        return []


@dataclass
class S2Data:
    s2_paper_id: str | None
    citation_count: int | None
    influential_citation_count: int | None
    tldr: str | None


async def enrich_papers(arxiv_ids: list[str]) -> dict[str, S2Data]:
    """arXiv ID 목록을 Semantic Scholar 배치 API로 보완.

    Returns:
        {arxiv_id: S2Data} — S2에 없는 논문은 결과에서 제외됨
    """
    if not arxiv_ids:
        return {}

    base_ids = [aid.split("v")[0] for aid in arxiv_ids]
    ids_payload = [f"arXiv:{aid}" for aid in base_ids]

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                _S2_BATCH_URL,
                params={"fields": _S2_BATCH_FIELDS},
                json={"ids": ids_payload},
                headers=_build_headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Semantic Scholar API 요청 한도 초과. 잠시 후 다시 시도해주세요.",
                )
            raise HTTPException(
                status_code=502,
                detail=f"Semantic Scholar API 오류: {e.response.status_code}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Semantic Scholar API 응답 시간 초과.")

    result: dict[str, S2Data] = {}
    for base_id, item in zip(base_ids, response.json()):
        if item is None:
            continue
        result[base_id] = S2Data(
            s2_paper_id=item.get("paperId"),
            citation_count=item.get("citationCount"),
            influential_citation_count=item.get("influentialCitationCount"),
            tldr=item.get("tldr", {}).get("text") if item.get("tldr") else None,
        )

    return result
