import asyncio
import logging
import time
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


# S2 키 한도: 모든 엔드포인트 합산 초당 1회 (2026-11 발급 메일). 워커가 하나라 프로세스 안의 잠금으로 충분하다.
# shortcut: 프로세스 단위 간격이라 blue/green이 겹치는 배포 중에는 두 컨테이너가 따로 센다 — 워커를 늘리면 공용 제한으로 바꿀 것
_MIN_INTERVAL_SECONDS = 1.1
_RETRY_DELAYS_SECONDS = (1.0, 2.0)  # 429면 간격을 늘려 재시도
_request_lock = asyncio.Lock()
_last_request_at = 0.0


async def _send(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """S2 요청을 직전 요청과 최소 간격을 두고 보내고, 429면 지수적으로 늘린 간격으로 재시도한다."""
    global _last_request_at
    for retry_delay in (*_RETRY_DELAYS_SECONDS, None):
        async with _request_lock:
            wait = _last_request_at + _MIN_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                response = await client.request(method, url, **kwargs)
            finally:
                _last_request_at = time.monotonic()
        if response.status_code != 429 or retry_delay is None:
            return response
        logger.warning(f"[S2] HTTP 429 — {retry_delay:.0f}초 뒤 재시도")
        await asyncio.sleep(retry_delay)
    raise AssertionError("unreachable")


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


async def search_papers(
    query: str,
    max_results: int = 5,
    date_from: str | None = None,
) -> list[dict]:
    """Semantic Scholar 검색 API로 논문을 검색한다.

    키 한도는 초당 1회(모든 엔드포인트 합산) — `_send`가 간격과 429 재시도를 맡는다.
    실패 시 빈 리스트를 반환해 arXiv fallback이 동작하도록 한다.

    Args:
        date_from: "YYYY-MM" 형식 — 이 날짜 이후 논문만 검색 (예: "2025-10")
    """
    params: dict = {
        "query": query,
        "limit": max_results,
        "fields": _S2_SEARCH_FIELDS,
    }
    if date_from:
        # S2 publicationDateOrYear: "2025-10:" → 2025년 10월 이후
        params["publicationDateOrYear"] = f"{date_from}:"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await _send(
                client,
                "GET",
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
            response = await _send(
                client,
                "POST",
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

    try:
        data = response.json()
    except Exception:
        logger.warning("[S2 Enrich] JSON 파싱 실패 — 빈 결과 반환")
        return {}

    result: dict[str, S2Data] = {}
    for base_id, item in zip(base_ids, data):
        if item is None:
            continue
        result[base_id] = S2Data(
            s2_paper_id=item.get("paperId"),
            citation_count=item.get("citationCount"),
            influential_citation_count=item.get("influentialCitationCount"),
            tldr=item.get("tldr", {}).get("text") if item.get("tldr") else None,
        )

    return result
