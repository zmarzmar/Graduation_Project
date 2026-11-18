import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationInfo, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_user, get_db, require_ai_enabled
from crud import analysis as crud_analysis
from crud import paper_document as crud_paper_document
from crud import qa_message as crud_qa_message
from models.user import User
from services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["qa"])


HISTORY_QUESTION_MAX = 500
HISTORY_ANSWER_MAX = 2000
# 모델에 함께 보내는 직전 대화 수 (2026-11-18: 2 → 6, 사용자 결정). 매 질문의 입력이 이만큼 늘어난다 — 저장 개수와는 별개다
HISTORY_TURNS = 6


class HistoryTurn(BaseModel):
    # 길면 거부하지 않고 자른다 — 긴 답변 뒤의 후속 질문이 매번 422로 막히지 않게 (Codex 리뷰 지적)
    question: str
    answer: str = ""

    @field_validator("question", "answer", mode="before")
    @classmethod
    def _trim(cls, value: object, info: ValidationInfo) -> str:
        limit = HISTORY_QUESTION_MAX if info.field_name == "question" else HISTORY_ANSWER_MAX
        return str(value or "").strip()[:limit]


class AskRequest(BaseModel):
    # 앞뒤 공백을 뗀 뒤에 길이를 잰다 — 공백뿐인 질문이 임베딩 호출까지 가지 않게 한다
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]
    # 직전 대화(최대 HISTORY_TURNS개) — 후속 질문("수식 알려줘")이 무엇을 가리키는지 알게 한다. 이 history 자체는 저장하지 않는다 (이번 질문과 답만 qa_messages에)
    history: Annotated[list[HistoryTurn], Field(max_length=HISTORY_TURNS)] = []


class CitationOut(BaseModel):
    page: int            # 원본 PDF 페이지 번호
    chunk_index: int
    quote: str           # 검색된 구절에 실제로 들어 있는 것이 확인된 문장


class ClaimOut(BaseModel):
    text: str                      # 답변의 한 문장
    citations: list[CitationOut]   # 이 문장에 달린 출처 — 모두 구절에 실재함이 확인됐다 (하나 이상)


class FormulaOut(BaseModel):
    name: str = ""
    latex: str
    description: str = ""


class WebSourceOut(BaseModel):
    title: str
    url: str


class WebAnswerOut(BaseModel):
    text: str                    # 웹 검색 일반 설명 — 서버가 내용을 확인하지 않았다 (논문 답변과 다르다)
    sources: list[WebSourceOut]  # 모델이 단 출처 링크 (http(s)만)


class AskResponse(BaseModel):
    answerable: bool     # False면 answer는 '근거를 확인하지 못함' 안내다
    answer: str          # claims의 문장을 이어 붙인 것
    claims: list[ClaimOut]  # 문장별 출처 연결. 근거를 확인하지 못했으면 빈 목록
    citations: list[CitationOut]  # 인용문이 구절에 실재함만 확인된 출처 — 문장을 뒷받침하는지는 확인하지 않았다
    dropped_citations: int  # 검증에서 제거된 출처 수
    dropped_claims: int     # 출처가 검증되지 않아 답변에서 뺀 문장 수 — 0보다 크면 답변이 모델이 쓴 것보다 짧다
    related_formulas: list[FormulaOut] = []  # 수식을 묻는 질문일 때 그 분석의 핵심 수식 — 원문 인용으로 확인한 것이 아니다
    web_answer: WebAnswerOut | None = None  # 논문에서 근거를 찾지 못했을 때만 — 웹 검색 일반 설명, 논문에서 확인한 내용이 아니다
    saved: bool = False  # 이 질문과 답을 분석 기록에 저장했는지 — False면 마이페이지에서 다시 볼 수 없다


class QaMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    response: dict  # 저장할 때의 AskResponse (saved 제외)
    created_at: datetime


@router.post(
    "/analyses/{analysis_id}/ask",
    response_model=AskResponse,
    responses={202: {"description": "색인 중 — Retry-After 뒤에 다시 요청"}, 409: {"description": "원문이 보관되지 않은 기록"},
               503: {"description": "관리자가 AI 기능을 껐다"}},
    dependencies=[Depends(require_ai_enabled)],
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

    key_formulas = await crud_analysis.get_key_formulas(db, analysis_id, current_user.id)
    # 색인·검색·답변은 외부 호출이다 — 요청의 DB 트랜잭션을 열어둔 채 기다리지 않는다
    await db.commit()

    try:
        result = await rag_service.ask(document, body.question, [h.model_dump() for h in body.history], key_formulas)
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

    # 응답 형식을 먼저 확인한 뒤 저장한다 — 화면에 못 내보낼 답을 저장하지 않게
    response = AskResponse(**result)
    try:
        stored = await crud_qa_message.save_for_active_analysis(
            db, analysis_id, current_user.id, body.question, response.model_dump(exclude={"saved"})
        )
        await db.commit()
    except Exception as e:
        # 저장이 실패해도 답은 보낸다 — 화면이 '저장되지 않았어요'를 보여준다
        await db.rollback()
        # 예외 문자열에는 INSERT 값(질문·답·인용문)이 들어갈 수 있다 — 종류만 남긴다 (Codex 리뷰)
        logger.error(f"[QA] 기록 저장 실패 analysis={analysis_id}: {type(e).__name__}")
        return response
    if not stored:
        # 답을 만드는 동안 다른 탭에서 분석 기록을 지웠다 — 질문 전과 같은 404
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return response.model_copy(update={"saved": True})


@router.get("/analyses/{analysis_id}/qa", response_model=list[QaMessageOut])
async def list_qa_messages(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """저장된 Q&A (최근 50개, 오래된 순). LLM을 부르지 않으므로 AI 스위치로 막지 않는다."""
    if not await crud_analysis.user_has_active_analysis(db, analysis_id, current_user.id):
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return await crud_qa_message.list_recent(db, analysis_id, current_user.id)
