import logging
import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
from fastapi import HTTPException

from schemas.paper import PaperResult

logger = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_HEADERS = {"User-Agent": "arxiv-analyst/0.1 (graduation-project; contact@example.com)"}

# 429 발생 시 재시도 설정
_MAX_RETRIES = 3
_RETRY_DELAYS = [3.0, 6.0, 12.0]  # 지수 백오프 (초)

# 인메모리 결과 캐시 — 같은 쿼리의 연속 요청 방지
_cache: dict[str, tuple[float, list[PaperResult]]] = {}  # {key: (timestamp, results)}
_CACHE_TTL = 300.0  # 5분


def _parse_entry(entry: feedparser.FeedParserDict) -> PaperResult:
    """feedparser 엔트리 → PaperResult 변환"""
    arxiv_id = entry.id.split("/abs/")[-1]

    pdf_url = next(
        (link.href for link in entry.get("links", []) if link.get("type") == "application/pdf"),
        f"https://arxiv.org/pdf/{arxiv_id}",
    )

    return PaperResult(
        arxiv_id=arxiv_id,
        title=entry.title.replace("\n", " ").strip(),
        authors=[a.name for a in entry.get("authors", [])],
        abstract=entry.summary.replace("\n", " ").strip(),
        url=entry.id,
        pdf_url=pdf_url,
        published_at=datetime(*entry.published_parsed[:6]),
        categories=[tag.term for tag in entry.get("tags", [])],
    )


async def search_papers(
    query: str,
    max_results: int = 10,
    recent_months: int | None = None,
) -> list[PaperResult]:
    """arXiv에서 논문 검색. 캐시 우선, 429 응답 시 최대 3회 자동 재시도.

    Args:
        recent_months: 설정 시 최근 N개월 이내 논문만 반환 (제출일 기준 필터)
    """
    cache_key = f"{query}:{max_results}:{recent_months}"
    now = time.monotonic()
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            logger.info(f"[arXiv] 캐시 히트 — '{query}' ({len(cached)}편)")
            return cached

    # recent_months 설정 시 최신순 정렬로 전환
    sort_by = "submittedDate" if recent_months else "relevance"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(_ARXIV_API_URL, params=params, headers=_HEADERS)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="arXiv API 요청 한도 초과. 잠시 후 다시 시도해주세요.",
                )
            raise HTTPException(
                status_code=502,
                detail=f"arXiv API 오류: {e.response.status_code}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="arXiv API 응답 시간 초과.")

    feed = feedparser.parse(response.text)
    results = [_parse_entry(entry) for entry in feed.entries]

    # recent_months 설정 시 날짜 기준 필터링
    if recent_months and results:
        cutoff = datetime.now(timezone.utc) - timedelta(days=recent_months * 30)
        results = [
            r for r in results
            if r.published_at and r.published_at.replace(tzinfo=timezone.utc) >= cutoff
        ]

    _cache[cache_key] = (now, results)
    return results
