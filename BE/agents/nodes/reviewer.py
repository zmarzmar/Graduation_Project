import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.state import AgentState
from core.config import settings

logger = logging.getLogger(__name__)

# o4-mini: 코드 추론 및 논문 대조 검증 특화
_llm = ChatOpenAI(
    model="o4-mini",
    api_key=settings.openai_api_key,
)

_SYSTEM_PROMPT = """당신은 AI 논문 구현 코드를 검토하는 전문 리뷰어입니다.
생성된 코드가 논문의 이론(알고리즘, 수식, 구조)을 올바르게 구현했는지 검토하고
아래 JSON 형식으로만 응답하세요.

{
  "passed": true 또는 false,
  "feedback": "전체 피드백 요약 (한국어)",
  "issues": ["문제점1", "문제점2"],
  "suggestions": ["개선 제안1", "개선 제안2"]
}

통과 기준 (관대하게 적용):
- 핵심 알고리즘의 전체적인 구조가 논문과 유사하면 통과
- 세부 파라미터가 완벽하지 않아도 주요 구성 요소가 있으면 통과
- 코드 스켈레톤 수준이므로 TODO 주석으로 처리된 부분은 문제로 보지 않음
- 실행 가능한 완전한 구조를 가지면 통과
- 사소한 구현 디테일 차이는 issues에 기록하되 passed는 true로 설정"""


def _extract_json(text: str) -> dict:
    """LLM 응답 텍스트에서 JSON 블록을 추출한다.
    순수 JSON 직접 파싱 → 코드 블록 → 브라켓 카운팅 순으로 시도한다.
    """
    # 1. LLM이 순수 JSON만 반환한 경우 바로 파싱
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` 코드 블록 추출 시도
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 브라켓 카운팅으로 첫 번째 완결된 JSON 객체 추출 — 문자열 내 {}도 올바르게 처리
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

    return {}


async def reviewer_node(state: AgentState) -> dict:
    """생성된 코드가 논문 이론을 올바르게 구현했는지 검증한다."""
    iteration = state.get("iteration_count", 0)
    logger.info(f"[Reviewer] 시작 — {iteration}회차 코드 검증")

    generated_code = state.get("generated_code", "")

    if not generated_code:
        logger.warning("[Reviewer] 검토할 코드 없음 — 낙제 처리")
        return {
            "review_feedback": "검토할 코드가 없습니다.",
            "review_passed": False,
            "current_node": "reviewer",
            "error": "generated_code가 비어 있습니다.",
        }

    # 논문 컨텍스트 — pdf_text가 있으면 전문 우선 사용 (Coder와 동일한 컨텍스트 보장)
    papers = state.get("papers", [])
    pdf_text = state.get("pdf_text", "")

    if pdf_text:
        # PDF 전문 앞 8000자 사용 — 토큰 한도를 고려하되 핵심 Methods 섹션 포함
        paper_context = f"[논문 전문 (앞 8000자)]\n{pdf_text[:8000]}"
    else:
        paper_lines = []
        for i, paper in enumerate(papers[:2], 1):
            paper_lines.append(f"### 논문 {i}: {paper.get('title', 'N/A')}")
            if tldr := paper.get("tldr"):
                paper_lines.append(f"요약: {tldr}")
            paper_lines.append(f"초록:\n{paper.get('abstract', '')[:600]}")
            paper_lines.append("")
        paper_context = "\n".join(paper_lines) or "논문 정보 없음"

    user_content = (
        f"### 참고 논문:\n{paper_context}\n\n"
        f"### 생성된 코드:\n```python\n{generated_code}\n```"
    )

    try:
        emit_log("reviewer", f"{iteration}회차 코드 검증 중...")
        emit_log("reviewer", "논문 이론과 코드 대조 분석 중...")
        response = await _llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        result = _extract_json(response.content)
    except Exception as e:
        logger.error(f"[Reviewer] LLM 호출 실패: {e}")
        emit_log("reviewer", f"검증 실패: {str(e)}")
        return {
            "review_feedback": f"리뷰 실패: {str(e)}",
            "review_passed": False,
            "current_node": "reviewer",
            "error": f"리뷰 중 오류 발생: {str(e)}",
        }

    passed: bool = result.get("passed", False)
    verdict = "✅ 통과" if passed else "❌ 낙제"
    emit_log("reviewer", verdict)
    if not passed and (issues := result.get("issues", [])):
        for issue in issues[:3]:
            emit_log("reviewer", f"문제점: {issue}")
    logger.info(f"[Reviewer] 완료 — {verdict} (iteration={iteration})")
    if not passed and (issues := result.get("issues", [])):
        for issue in issues[:3]:
            logger.info(f"[Reviewer]   문제점: {issue[:80]}")

    # 피드백 텍스트 조합
    feedback_parts = [result.get("feedback", "")]
    if issues := result.get("issues", []):
        feedback_parts.append("문제점:\n" + "\n".join(f"- {i}" for i in issues))
    if suggestions := result.get("suggestions", []):
        feedback_parts.append("개선 제안:\n" + "\n".join(f"- {s}" for s in suggestions))
    feedback = "\n\n".join(filter(None, feedback_parts))

    # 최종 결과물 구성
    final_result = {
        "papers": papers,
        "generated_code": generated_code,
        "review_feedback": feedback,
        "review_passed": passed,
        "iteration_count": state.get("iteration_count", 0),
    }

    return {
        "review_feedback": feedback,
        "review_passed": passed,
        "final_result": final_result,
        "current_node": "reviewer",
        "error": None,
    }
