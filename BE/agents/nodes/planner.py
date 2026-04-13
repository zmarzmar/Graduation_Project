import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.state import AgentState
from core.config import settings

logger = logging.getLogger(__name__)

# JSON 응답 강제 — gpt-4o-mini는 response_format 지원
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_SYSTEM_PROMPT = """당신은 AI 논문 분석 시스템의 플래너입니다.
사용자 요청을 분석하고 아래 JSON 형식으로만 실행 계획을 반환하세요.

{
  "summary": "실행 계획 요약 (한국어, 2-3문장)",
  "search_keywords": ["영어 키워드1", "영어 키워드2", "영어 키워드3"],
  "focus_area": "분석 초점 (예: transformer architecture, LoRA fine-tuning)",
  "framework": "pytorch 또는 tensorflow"
}

규칙:
- search_keywords는 arXiv 검색에 최적화된 영어 키워드로 작성
- framework는 논문에서 사용한 것으로 추정하고, 불명확하면 "pytorch" 사용
- trend 모드면 최신 트렌드를 반영한 영어 키워드 포함"""


async def planner_node(state: AgentState) -> dict:
    """사용자 입력을 분석하고 실행 계획을 수립한다."""
    mode = state["mode"]
    logger.info(f"[Planner] 시작 — mode={mode}, query=\"{state['user_query']}\"")

    if mode == "pdf":
        user_content = (
            "모드: PDF 업로드\n\n"
            f"PDF 내용 (앞 3000자):\n{state.get('pdf_text', '')[:3000]}"
        )
    elif mode == "analyze":
        # 사용자가 선택한 논문 1편 분석 — pdf_text 또는 초록으로 키워드 추출
        pdf_text = state.get("pdf_text", "")
        papers = state.get("papers", [])
        if pdf_text:
            context = f"PDF 내용 (앞 3000자):\n{pdf_text[:3000]}"
        elif papers:
            first = papers[0]
            context = f"논문 제목: {first.get('title', '')}\n초록: {first.get('abstract', '')[:1000]}"
        else:
            context = f"검색어: {state['user_query']}"
        user_content = f"모드: 논문 선택 분석\n\n{context}"
    elif mode == "search":
        user_content = f"모드: 키워드 검색\n\n검색어: {state['user_query']}"
    else:  # trend
        user_content = f"모드: 트렌드 브리핑\n\n관심 분야: {state['user_query']}"

    try:
        emit_log("planner", "요청 분석 중...")
        response = await _llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        plan = json.loads(response.content)
        keywords = plan.get('search_keywords', [])
        emit_log("planner", f"검색 키워드: {', '.join(keywords)}")
        logger.info(f"[Planner] 완료 — 키워드: {keywords}")
    except Exception as e:
        logger.error(f"[Planner] LLM 호출 실패: {e}")
        emit_log("planner", "계획 수립 실패 — 기본값으로 진행")
        # LLM 실패 시 사용자 쿼리를 그대로 폴백으로 사용
        plan = {
            "summary": f"{state['user_query']} 관련 논문을 분석합니다.",
            "search_keywords": [state["user_query"]],
            "focus_area": state["user_query"],
            "framework": "pytorch",
        }

    return {
        "plan": json.dumps(plan, ensure_ascii=False),
        "current_node": "planner",
        "error": None,
    }
