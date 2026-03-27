import logging

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_OA_WORKS_URL = "https://api.openalex.org/works"
_OA_SEARCH_FIELDS = "id,doi,title,authorships,abstract_inverted_index,publication_year,cited_by_count,open_access,primary_location,ids"


def _build_params(query: str, max_results: int) -> dict:
    """OpenAlex 검색 파라미터를 구성한다.
    이메일이 설정된 경우 polite pool(100 req/sec)을 사용한다.
    """
    params: dict = {
        "search": query,
        "per-page": max_results,
        "select": _OA_SEARCH_FIELDS,
    }
    if settings.openalex_email:
        params["mailto"] = settings.openalex_email
    return params


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex의 역색인 형식 초록을 일반 문자열로 복원한다."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def _oa_item_to_dict(item: dict) -> dict:
    """OpenAlex works 항목 1건을 papers state 형식의 dict로 변환한다."""
    ids = item.get("ids") or {}
    # arXiv ID 추출 — "https://arxiv.org/abs/2301.00001" 형태에서 ID만 파싱
    arxiv_raw = ids.get("arxiv", "")
    arxiv_id = arxiv_raw.split("/abs/")[-1] if "/abs/" in arxiv_raw else ""

    year = item.get("publication_year") or 2024

    # PDF URL: primary_location → open_access 순으로 탐색
    pdf_url = ""
    if pl := item.get("primary_location"):
        pdf_url = pl.get("pdf_url") or ""
    if not pdf_url:
        if oa := item.get("open_access"):
            pdf_url = oa.get("oa_url") or ""
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    # 논문 URL
    url = ""
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi := item.get("doi"):
        url = doi  # doi 필드가 전체 URL 형태로 제공됨

    authors = [
        auth.get("author", {}).get("display_name", "")
        for auth in (item.get("authorships") or [])
    ]

    return {
        "arxiv_id": arxiv_id,
        "title": item.get("title") or "",
        "authors": authors,
        "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),
        "url": url,
        "pdf_url": pdf_url,
        "published_at": f"{year}-01-01T00:00:00",
        "categories": [],
        "citation_count": item.get("cited_by_count"),
        "influential_citation_count": None,
        "tldr": None,
        "s2_paper_id": None,
    }


async def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """OpenAlex Works 검색 API로 논문을 검색한다.

    완전 무료, API 키 불필요. 이메일을 mailto 파라미터로 넘기면
    polite pool(100 req/sec)로 자동 전환된다.
    실패 시 빈 리스트를 반환해 다음 fallback이 동작하도록 한다.
    """
    params = _build_params(query, max_results)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                _OA_WORKS_URL,
                params=params,
                headers={"User-Agent": "arxiv-analyst/0.1 (graduation-project; contact@example.com)"},
            )
            response.raise_for_status()
        items = response.json().get("results", [])
        results = [_oa_item_to_dict(item) for item in items if item.get("title")]
        logger.info(f"[OpenAlex] '{query}' — {len(results)}편")
        return results
    except httpx.HTTPStatusError as e:
        logger.warning(f"[OpenAlex] HTTP {e.response.status_code} — 다음 소스로 fallback")
        return []
    except Exception as e:
        logger.warning(f"[OpenAlex] 실패: {e} — 다음 소스로 fallback")
        return []
