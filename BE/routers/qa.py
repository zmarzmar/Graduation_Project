import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_user, get_db
from crud import analysis as crud_analysis
from crud import paper_document as crud_paper_document
from models.user import User
from services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["qa"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)


class CitationOut(BaseModel):
    page: int            # 원본 PDF 페이지 번호
    chunk_index: int
    quote: str           # 검색된 구절에 실제로 들어 있는 것이 확인된 문장


class AskResponse(BaseModel):
    answerable: bool     # False면 answer는 '근거를 확인하지 못함' 안내다
    answer: str
    citations: list[CitationOut]
    dropped_citations: int  # 검증에서 제거된 출처 수 — 0보다 크면 답변에 근거 없는 문장이 남았을 수 있다


@router.post(
    "/analyses/{analysis_id}/ask",
    response_model=AskResponse,
    responses={202: {"description": "색인 중 — Retry-After 뒤에 다시 요청"}, 409: {"description": "원문이 보관되지 않은 기록"}},
)
async def ask_about_analyzed_paper(
    analysis_id: int,
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),  # 로그인 전용 — 게스트는 원문을 보관하지 않는다
):
    """분석한 논문 한 편에 질문한다. 첫 질문은 색인을 만드느라 더 오래 걸린다."""
    document = await crud_paper_document.get_document_for_analysis(db, analysis_id, current_user.id)
    if document is None:
        # 기록 자체가 없는 것(404)과, 기록은 있는데 원문이 없는 것(409)을 구분해서 알린다
        if not await crud_analysis.user_has_active_analysis(db, analysis_id, current_user.id):
            raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
        return JSONResponse(
            status_code=409,
            content={
                "detail": "이 기록에는 보관된 논문 원문이 없어 질문할 수 없습니다. 논문을 다시 분석하면 사용할 수 있습니다.",
                "reason": "no_document",
            },
        )

    # 색인·검색·답변은 외부 호출이다 — 요청의 DB 트랜잭션을 열어둔 채 기다리지 않는다
    await db.commit()

    try:
        return await rag_service.ask(document, body.question.strip())
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
        logger.error(f"[QA] 질문 처리 실패 analysis={analysis_id}: {e}")
        raise HTTPException(status_code=502, detail="질문을 처리하지 못했습니다. 잠시 후 다시 시도해주세요.")
