from datetime import datetime

from pydantic import BaseModel


class PaperResult(BaseModel):
    # arXiv 기본 필드
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str
    published_at: datetime
    categories: list[str]
    # Semantic Scholar 보완 필드
    citation_count: int | None = None
    influential_citation_count: int | None = None
    tldr: str | None = None
    s2_paper_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    total: int
    papers: list[PaperResult]


class TrendTopicsBasis(BaseModel):
    label: str
    source: str
    observed_from: str
    observed_to: str
    updated_at: str
    paper_count: int


class DailyKeywordsResponse(BaseModel):
    keywords: list[str]
    basis: TrendTopicsBasis | None  # None이면 근거 없는 일반 검색 예시
