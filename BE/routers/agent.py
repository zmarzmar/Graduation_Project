from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services import agent_service

router = APIRouter(tags=["agent"])


# ── 요청 모델 ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="분석할 논문 키워드")


class TrendRequest(BaseModel):
    topic: str = Field(
        default="large language model",
        description="트렌드를 조회할 AI 분야 (예: 'vision transformer', 'RL')",
    )


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.post("/agent/pdf")
async def run_pdf_agent(file: UploadFile = File(...)) -> StreamingResponse:
    """PDF를 업로드하면 논문을 자동 분석하고 코드를 재현한다. (SSE 스트리밍)"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        pdf_text = agent_service.extract_pdf_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF 텍스트 추출 실패: {str(e)}")

    return StreamingResponse(
        agent_service.stream_agent(
            mode="pdf",
            user_query=file.filename,
            pdf_text=pdf_text,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},  # Nginx 버퍼링 비활성화
    )


@router.post("/agent/search")
async def run_search_agent(request: SearchRequest) -> StreamingResponse:
    """키워드로 arXiv 논문을 검색하고 분석 및 코드를 재현한다. (SSE 스트리밍)"""
    return StreamingResponse(
        agent_service.stream_agent(
            mode="search",
            user_query=request.query,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/agent/trend")
async def run_trend_agent(request: TrendRequest) -> StreamingResponse:
    """HuggingFace + arXiv 기반 최신 트렌드 논문 요약 리포트를 생성한다. (SSE 스트리밍)"""
    return StreamingResponse(
        agent_service.stream_agent(
            mode="trend",
            user_query=request.topic,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
