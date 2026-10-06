import logging
from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_user, get_db
from crud import analysis as crud_analysis
from crud import paper_document as crud_paper_document
from models.paper_document import PaperDocument
from models.user import User
from services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["qa"])


class AskRequest(BaseModel):
    # 앞뒤 공백을 뗀 뒤에 길이를 잰다 — 공백뿐인 질문이 임베딩 호출까지 가지 않게 한다
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]


class CitationOut(BaseModel):
    page: int            # 원본 PDF 페이지 번호
    chunk_index: int
    quote: str           # 검색된 구절에 실제로 들어 있는 것이 확인된 문장


class AskResponse(BaseModel):
    answerable: bool     # False면 answer는 '근거를 확인하지 못함' 안내다
    answer: str
    citations: list[CitationOut]  # 인용문이 구절에 실재함만 확인된 출처 — 문장을 뒷받침하는지는 확인하지 않았다
    dropped_citations: int  # 검증에서 제거된 출처 수
    dropped_claims: int     # 출처가 검증되지 않아 답변에서 뺀 문장 수 — 0보다 크면 답변이 모델이 쓴 것보다 짧다


class RelatedPassageOut(BaseModel):
    page: int            # 원본 PDF 페이지 번호
    chunk_index: int
    text: str            # 검색된 구절 원문


class RelatedItemOut(BaseModel):
    id: str              # summary-0, formula-1 … 저장된 분석 안에서의 순번
    kind: str            # summary | formula
    label: str           # 요약 문장 또는 수식 이름
    passages: list[RelatedPassageOut]


class RelatedResponse(BaseModel):
    # 유사도 검색 결과다 — 구절이 항목을 뒷받침하는지는 확인하지 않았다 (Q&A의 '확인된 인용'과 다르다)
    items: list[RelatedItemOut]


_INDEXING_RESPONSES = {
    202: {"description": "색인 중 — retry_after_seconds 뒤에 다시 요청"},
    409: {"description": "원문이 보관되지 않은 기록"},
}


async def _stored_document(db: AsyncSession, analysis_id: int, user_id: int) -> PaperDocument | JSONResponse:
    """질문·검색에 쓸 문서를 가져온다. 기록 자체가 없는 것(404)과 기록은 있는데 원문이 없는 것(409)을 구분해서 알린다."""
    document = await crud_paper_document.get_document_for_analysis(db, analysis_id, user_id)
    if document is not None:
        return document
    if not await crud_analysis.user_has_active_analysis(db, analysis_id, user_id):
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return JSONResponse(
        status_code=409,
        content={
            "detail": "이 기록에는 보관된 논문 원문이 없어 질문할 수 없습니다. 논문을 다시 분석하면 사용할 수 있습니다.",
            "reason": "no_document",
        },
    )


async def _run_search(work: Awaitable[dict], what: str) -> dict | JSONResponse:
    """색인·검색을 실행하고 색인 중(202)·시간 초과(504)·외부 호출 실패(502)를 응답으로 옮긴다."""
    try:
        return await work
    except rag_service.IndexingInProgress:
        return JSONResponse(
            status_code=202,
            headers={"Retry-After": str(rag_service.RETRY_AFTER_SECONDS)},
            content={"status": "indexing", "retry_after_seconds": rag_service.RETRY_AFTER_SECONDS,
                     "detail": "논문 검색을 준비하는 중입니다."},
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="논문 검색 준비가 시간 안에 끝나지 않았습니다. 잠시 후 다시 시도해주세요.")
    except Exception as e:
        logger.error(f"[QA] {what} 실패: {e}")
        raise HTTPException(status_code=502, detail="질문을 처리하지 못했습니다. 잠시 후 다시 시도해주세요.")


@router.post("/analyses/{analysis_id}/ask", response_model=AskResponse, responses=_INDEXING_RESPONSES)
async def ask_about_analyzed_paper(
    analysis_id: int,
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),  # 로그인 전용 — 게스트는 원문을 보관하지 않는다
):
    """분석한 논문 한 편에 질문한다. 첫 질문은 색인을 만드느라 더 오래 걸린다."""
    document = await _stored_document(db, analysis_id, current_user.id)
    if isinstance(document, JSONResponse):
        return document
    # 색인·검색·답변은 외부 호출이다 — 요청의 DB 트랜잭션을 열어둔 채 기다리지 않는다
    await db.commit()
    return await _run_search(rag_service.ask(document, body.question), f"질문 처리 analysis={analysis_id}")


@router.get("/analyses/{analysis_id}/related", response_model=RelatedResponse, responses=_INDEXING_RESPONSES)
async def related_passages_of_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """분석 결과의 요약 문장·핵심 수식마다 관련 원문 구절을 찾는다 (유사도 검색 — 뒷받침 여부는 확인하지 않는다).

    질의는 서버에 저장된 분석에서 만든다. 클라이언트는 검색어를 보내지 않는다.
    """
    document = await _stored_document(db, analysis_id, current_user.id)
    if isinstance(document, JSONResponse):
        return document
    analysis = await crud_analysis.get_active_analysis(db, analysis_id, current_user.id)
    if analysis is None:  # 문서를 확인한 직후에 기록이 삭제됐다
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    summary, formulas = analysis.paper_summary, analysis.key_formulas
    await db.commit()

    async def search() -> dict:
        return {"items": await rag_service.related_passages(document, summary, formulas)}

    return await _run_search(search(), f"관련 원문 검색 analysis={analysis_id}")
