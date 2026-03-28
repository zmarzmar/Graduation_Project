import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.state import AgentState
from core.config import settings

logger = logging.getLogger(__name__)

# o4-mini: 코드 추론 특화 모델 — temperature 파라미터 미지정
_llm = ChatOpenAI(
    model="o4-mini",
    api_key=settings.openai_api_key,
)

_SYSTEM_PROMPT = """당신은 AI 논문을 PyTorch 코드로 구현하는 전문가입니다.
논문의 Methods 섹션을 분석해 실행 가능한 PyTorch 코드 스켈레톤을 작성하세요.

코드 작성 규칙:
- PyTorch 기반 작성 (논문이 TensorFlow를 명시한 경우 TensorFlow 사용)
- 모든 클래스와 함수에 타입 힌트 필수
- 논문의 핵심 알고리즘/수식을 인라인 주석으로 표기
- 구현이 필요한 부분은 TODO 주석으로 명시
- import 포함, 실행 가능한 완전한 코드로 작성"""


def _build_paper_context(papers: list[dict]) -> str:
    """논문 목록을 프롬프트용 컨텍스트 텍스트로 변환한다. 최대 3편만 사용."""
    if not papers:
        return "수집된 논문 없음"

    lines = []
    for i, paper in enumerate(papers[:3], 1):
        lines.append(f"## 논문 {i}: {paper.get('title', 'N/A')}")
        lines.append(f"저자: {', '.join(paper.get('authors', [])[:3])}")
        if tldr := paper.get("tldr"):
            lines.append(f"요약: {tldr}")
        lines.append(f"초록:\n{paper.get('abstract', '')[:800]}")
        lines.append("")
    return "\n".join(lines)


async def coder_node(state: AgentState) -> dict:
    """논문 내용을 분석해 PyTorch 코드 스켈레톤을 생성한다.
    2회차 이상에서는 review_feedback를 반영해 코드를 수정한다.
    """
    paper_context = _build_paper_context(state.get("papers", []))
    iteration = state.get("iteration_count", 0)
    label = "최초 생성" if iteration == 0 else f"피드백 반영 ({iteration}회차 → {iteration + 1}회차)"
    logger.info(f"[Coder] 시작 — {label}")

    # Analyzer 결과 컨텍스트 추가
    summary = state.get("paper_summary", "")
    key_formulas = state.get("key_formulas", [])
    formula_context = ""
    if key_formulas:
        formula_lines = ["### 핵심 수식:"]
        for f in key_formulas:
            formula_lines.append(f"- {f.get('name', '')}: ${f.get('latex', '')}$")
            formula_lines.append(f"  ({f.get('description', '')})")
        formula_context = "\n".join(formula_lines)

    if iteration == 0:
        user_content = (
            "다음 논문을 바탕으로 PyTorch 코드 스켈레톤을 작성하세요.\n\n"
            + (f"### 논문 요약:\n{summary}\n\n" if summary else "")
            + (f"{formula_context}\n\n" if formula_context else "")
            + f"{paper_context}"
        )
    else:
        user_content = (
            "리뷰어 피드백을 반영해 이전 코드를 수정하세요.\n\n"
            f"### 리뷰어 피드백:\n{state.get('review_feedback', '')}\n\n"
            f"### 이전 코드:\n{state.get('generated_code', '')}\n\n"
            f"### 참고 논문:\n{paper_context}"
        )

    try:
        paper_count = len(state.get("papers", []))
        emit_log("coder", f"논문 {min(paper_count, 3)}편 분석 중...")
        if iteration > 0:
            emit_log("coder", f"리뷰어 피드백 반영 ({iteration}회차 수정)")
        emit_log("coder", "PyTorch 코드 생성 중...")
        response = await _llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        generated_code = response.content
        emit_log("coder", f"코드 생성 완료 ({len(generated_code)}자)")
        logger.info(f"[Coder] 완료 — 코드 {len(generated_code)}자 생성")
    except Exception as e:
        logger.error(f"[Coder] LLM 호출 실패: {e}")
        emit_log("coder", f"코드 생성 실패: {str(e)}")
        return {
            "generated_code": "",
            "iteration_count": iteration + 1,
            "current_node": "coder",
            "error": f"코드 생성 실패: {str(e)}",
        }

    return {
        "generated_code": generated_code,
        "iteration_count": iteration + 1,
        "current_node": "coder",
        "error": None,
    }
