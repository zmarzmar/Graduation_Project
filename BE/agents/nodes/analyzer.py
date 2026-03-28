import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.state import AgentState
from core.config import settings

logger = logging.getLogger(__name__)

# gpt-4o-mini: 요약/분석 작업
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_SYSTEM_PROMPT = """당신은 AI 논문을 분석하는 전문가입니다.
주어진 논문을 분석하고 아래 JSON 형식으로만 응답하세요.

{
  "summary": "논문 핵심 요약 (한국어, 3-5문장. 연구 목적, 방법, 결과, 기여 포함)",
  "review": {
    "strengths": ["강점1", "강점2", "강점3"],
    "limitations": ["한계점1", "한계점2"],
    "significance": "이 논문의 학문적/실용적 의의 (한국어, 2-3문장)"
  },
  "key_formulas": [
    {
      "name": "수식 이름 (예: Attention Score)",
      "latex": "LaTeX 수식 (예: \\\\text{Attention}(Q,K,V) = \\\\text{softmax}(\\\\frac{QK^T}{\\\\sqrt{d_k}})V)",
      "description": "이 수식이 하는 역할 설명 (한국어, 1-2문장)"
    }
  ]
}

규칙:
- key_formulas는 논문에서 핵심적인 수식/알고리즘을 최대 3개까지만 추출
- 수식이 없는 논문이면 key_formulas는 빈 배열로
- LaTeX 수식은 인라인 수식 형태로 작성"""


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON을 추출한다."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


async def analyzer_node(state: AgentState) -> dict:
    """논문 요약, 리뷰, 핵심 수식을 추출한다. (pdf/search 모드 전용)"""
    logger.info("[Analyzer] 시작")

    papers = state.get("papers", [])
    pdf_text = state.get("pdf_text", "")

    # 분석 대상 텍스트 구성 — PDF 텍스트 우선, 없으면 수집된 논문 사용
    if pdf_text:
        paper_text = pdf_text[:6000]
        emit_log("analyzer", "PDF 내용 분석 중...")
    elif papers:
        first = papers[0]
        paper_text = f"제목: {first.get('title', '')}\n\n초록:\n{first.get('abstract', '')}"
        if tldr := first.get("tldr"):
            paper_text += f"\n\nTL;DR: {tldr}"
        emit_log("analyzer", f"논문 분석 중: {first.get('title', '')[:50]}...")
    else:
        logger.warning("[Analyzer] 분석할 논문 없음")
        return {
            "paper_summary": "",
            "paper_review": {},
            "key_formulas": [],
            "current_node": "analyzer",
            "error": "분석할 논문이 없습니다.",
        }

    try:
        emit_log("analyzer", "요약 및 리뷰 생성 중...")
        response = await _llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"다음 논문을 분석해주세요:\n\n{paper_text}"),
        ])
        result = json.loads(response.content)
    except Exception as e:
        logger.error(f"[Analyzer] LLM 호출 실패: {e}")
        try:
            result = _extract_json(response.content if 'response' in dir() else "")
        except Exception:
            result = {}

    summary = result.get("summary", "")
    review = result.get("review", {})
    key_formulas = result.get("key_formulas", [])

    emit_log("analyzer", "요약 완료")
    if key_formulas:
        emit_log("analyzer", f"핵심 수식 {len(key_formulas)}개 추출 완료")
    logger.info(f"[Analyzer] 완료 — 수식 {len(key_formulas)}개 추출")

    return {
        "paper_summary": summary,
        "paper_review": review,
        "key_formulas": key_formulas,
        "current_node": "analyzer",
        "error": None,
    }
