import asyncio
import logging
import time
from datetime import datetime

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


async def search_papers(query: str, max_results: int = 10) -> list[PaperResult]:
    """arXiv에서 논문 검색. 캐시 우선, 429 응답 시 최대 3회 자동 재시도."""
    cache_key = f"{query}:{max_results}"
    now = time.monotonic()
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            logger.info(f"[arXiv] 캐시 히트 — '{query}' ({len(cached)}편)")
            return cached

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.get(_ARXIV_API_URL, params=params, headers=_HEADERS)
                response.raise_for_status()
                break  # 성공 시 루프 탈출

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_DELAYS[attempt]
                        logger.warning(
                            f"[arXiv] 429 rate limit — {delay:.0f}초 후 재시도 "
                            f"({attempt + 1}/{_MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise HTTPException(
                        status_code=429,
                        detail="arXiv API 요청 한도 초과. 잠시 후 다시 시도해주세요.",
                    )
                raise HTTPException(
                    status_code=502,
                    detail=f"arXiv API 오류: {e.response.status_code}",
                )

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(f"[arXiv] 타임아웃 — {delay:.0f}초 후 재시도 ({attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                raise HTTPException(status_code=504, detail="arXiv API 응답 시간 초과.")

    feed = feedparser.parse(response.text)
    results = [_parse_entry(entry) for entry in feed.entries]
    _cache[cache_key] = (now, results)
    return results
