import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.state import AgentState
from core.config import settings

logger = logging.getLogger(__name__)

# 요약·키워드 추출은 복잡한 추론 불필요 — gpt-4o-mini로 비용 절감
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_SYSTEM_PROMPT = """당신은 AI 논문 트렌드 분석 전문가입니다.
최신 AI 논문 목록을 분석하고 아래 JSON 형식으로만 응답하세요.

{
  "paper_summaries": [
    {
      "title": "논문 제목 (원문 그대로)",
      "summary": "핵심 기여와 방법론을 담은 한 문장 요약 (한국어, 50자 이내)"
    }
  ],
  "trending_keywords": [
    {
      "keyword": "트렌드 키워드 (영어)",
      "explanation": "이 키워드가 주목받는 이유 (한국어, 40자 이내)"
    }
  ]
}

규칙:
- paper_summaries는 입력된 모든 논문 각각에 대해 작성
- trending_keywords는 논문들에서 공통으로 나타나는 주요 주제 3~5개 추출
- summary는 기술적 내용을 쉽게 풀어 작성
- 초록이 없으면 제목만으로 최선의 요약 작성"""


def _sanitize(text: str) -> str:
    """null 바이트 및 제어 문자를 제거해 JSON 직렬화 오류를 방지한다."""
    return "".join(ch for ch in text if ch >= " " or ch in "\n\t")


def _build_prompt(papers: list[dict]) -> str:
    """논문 목록을 프롬프트용 텍스트로 변환한다."""
    lines = []
    for i, paper in enumerate(papers, 1):
        lines.append(f"## 논문 {i}")
        lines.append(f"제목: {_sanitize(paper.get('title', 'N/A'))}")
        if abstract := paper.get("abstract", ""):
            lines.append(f"초록: {_sanitize(abstract)[:400]}")
        lines.append("")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON을 추출한다."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def trend_analyzer_node(state: AgentState) -> dict:
    """수집된 트렌드 논문을 AI로 분석해 한 줄 요약과 트렌드 키워드를 생성한다."""
    papers = state.get("papers", [])
    logger.info(f"[TrendAnalyzer] 시작 — 논문 {len(papers)}편 분석")

    if not papers:
        logger.warning("[TrendAnalyzer] 분석할 논문 없음")
        empty = {"paper_summaries": [], "trending_keywords": []}
        return {
            "trend_analysis": empty,
            "final_result": {"papers": [], "trend_analysis": empty, "mode": "trend"},
            "current_node": "trend_analyzer",
            "error": "수집된 논문이 없어 분석을 건너뜁니다.",
        }

    emit_log("trend_analyzer", f"논문 {len(papers)}편 분석 중...")
    emit_log("trend_analyzer", "각 논문 한 줄 요약 생성 중...")

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_prompt(papers)),
        ])
        result = _extract_json(response.content)
    except Exception as e:
        logger.error(f"[TrendAnalyzer] LLM 호출 실패: {e}")
        emit_log("trend_analyzer", f"분석 실패: {str(e)}")
        empty = {"paper_summaries": [], "trending_keywords": []}
        return {
            "trend_analysis": empty,
            "final_result": {"papers": papers, "trend_analysis": empty, "mode": "trend"},
            "current_node": "trend_analyzer",
            "error": f"트렌드 분석 실패: {str(e)}",
        }

    summaries = result.get("paper_summaries", [])
    keywords = result.get("trending_keywords", [])

    emit_log("trend_analyzer", f"한 줄 요약 {len(summaries)}편 완료")
    emit_log("trend_analyzer", f"트렌드 키워드 {len(keywords)}개 추출")
    for kw in keywords:
        emit_log("trend_analyzer", f"🔑 {kw.get('keyword', '')} — {kw.get('explanation', '')}")

    logger.info(f"[TrendAnalyzer] 완료 — 요약 {len(summaries)}편, 키워드 {len(keywords)}개")

    trend_analysis = {"paper_summaries": summaries, "trending_keywords": keywords}
    final_result = {
        "papers": papers,
        "trend_analysis": trend_analysis,
        "mode": "trend",
    }

    return {
        "trend_analysis": trend_analysis,
        "final_result": final_result,
        "current_node": "trend_analyzer",
        "error": None,
    }
