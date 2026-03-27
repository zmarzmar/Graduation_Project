from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.paper import Paper
from schemas.paper import PaperResult


async def get_paper_by_arxiv_id(db: AsyncSession, arxiv_id: str) -> Paper | None:
    """arxiv_id로 논문 조회"""
    result = await db.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
    return result.scalar_one_or_none()


async def upsert_paper(db: AsyncSession, paper_data: PaperResult) -> Paper:
    """논문 저장 — 이미 존재하면 업데이트, 없으면 삽입"""
    existing = await get_paper_by_arxiv_id(db, paper_data.arxiv_id)
    if existing:
        existing.title = paper_data.title
        existing.authors = paper_data.authors
        existing.abstract = paper_data.abstract
        existing.citation_count = paper_data.citation_count
        existing.tldr = paper_data.tldr
        return existing

    paper = Paper(
        arxiv_id=paper_data.arxiv_id,
        title=paper_data.title,
        authors=paper_data.authors,
        abstract=paper_data.abstract,
        url=paper_data.url,
        pdf_url=paper_data.pdf_url,
        published_at=paper_data.published_at,
        categories=paper_data.categories,
        citation_count=paper_data.citation_count,
        tldr=paper_data.tldr,
    )
    db.add(paper)
    await db.flush()  # id 생성
    return paper
