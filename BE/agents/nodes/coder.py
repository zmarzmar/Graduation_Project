import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.exec_check import SMOKE_RULES, fixed_test_notice
from agents.log_stream import emit_log
from agents.paper_context import paper_source_context
from agents.perf import log_elapsed
from agents.state import AgentState
from agents.token_budget import REASONING_OUTPUT_BUDGET, ensure_complete, ensure_request_fits
from core.config import settings

logger = logging.getLogger(__name__)

# o4-mini가 폐기돼(2026-10, 404) 교체. 예전 o4-mini처럼 추론한 뒤 답하도록 강도를 medium으로 둔다 (gpt-5.4-mini 기본값은 none)
_MODEL = "gpt-5.4-mini"
_llm = ChatOpenAI(
    model=_MODEL,
    api_key=settings.openai_api_key,
    reasoning_effort="medium",
    max_tokens=REASONING_OUTPUT_BUDGET,  # 출력 예산 (추론 토큰 포함)
)

_SYSTEM_PROMPT = """당신은 AI 논문을 PyTorch 코드로 구현하는 전문가입니다.
논문의 Methods 섹션을 분석해 실행 가능한 PyTorch 코드 스켈레톤을 작성하세요.

코드 작성 규칙:
- PyTorch 기반 작성 (논문이 TensorFlow를 명시한 경우 TensorFlow 사용)
- 모든 클래스와 함수에 타입 힌트 필수
- 논문의 핵심 알고리즘/수식을 인라인 주석으로 표기
- 구현이 필요한 부분은 TODO 주석으로 명시
- import 포함, 실행 가능한 완전한 코드로 작성
- 설명이 필요하면 반드시 Python 주석(#)으로만 작성할 것""" + SMOKE_RULES


# 구현 대상이 아닌 논문의 표시 — 최초 생성 요청에만 이 선택지를 준다 (수정 회차에서 코드를 버리지 않게)
_SKIP_MARKER = "NOT_IMPLEMENTABLE:"
_SKIP_INSTRUCTION = (
    "먼저 이 논문이 스스로 제안하거나 구체적으로 정의한 모델·모듈·알고리즘·학습/평가 절차가 있는지 판단하세요 "
    "(이론 논문이 증명에서 구체적으로 구성한 구조도 포함). 다음 경우에는 코드를 쓰지 말고 "
    f"첫 줄에 `{_SKIP_MARKER} <이유 한두 문장, 한국어>`만 쓰세요.\n"
    "- 서베이·리뷰: 다른 논문들의 방법을 정리한 논문. 그 방법들을 자세히 설명하더라도 논문이 제안한 방법이 아니므로, 일부를 골라 구현하지 않는다\n"
    "- 연구 방향·문제 제기·입장 논문: 구체적인 방법 없이 과제나 방향을 제시하는 논문. 여러 아이디어를 엮어 새 시스템을 지어내지 않는다\n"
    "- 구체적인 구성이 없는 순수 이론·증명 논문\n"
    "위 경우가 아니고 논문이 제안한 방법이 있으면 그 방법을 구현하세요.\n\n"
)


def _skip_reason(text: str) -> str | None:
    """응답의 첫 줄(코드 블록 표시 제외)이 구현 대상 아님 표시면 그 이유를, 아니면 None을 돌려준다."""
    for line in text.splitlines():
        line = line.strip()
        # 빈 줄과 코드 블록 시작 줄(```, ```python)은 건너뛴다
        if not line or line.startswith("```"):
            continue
        line = line.strip("`").strip()  # 지시문의 예시처럼 한 줄을 백틱으로 감싼 경우
        if line.startswith(_SKIP_MARKER):
            return line[len(_SKIP_MARKER):].strip() or "구현할 방법을 찾지 못했습니다."
        return None
    return None


def _extract_code(text: str) -> str:
    """LLM 응답에서 Python 코드 블록만 추출한다.
    ```python ... ``` 블록이 있으면 그 내용만 반환하고,
    없으면 전체 텍스트를 그대로 반환한다.
    """
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def coder_node(state: AgentState) -> dict:
    """논문 내용을 분석해 PyTorch 코드 스켈레톤을 생성한다.
    2회차 이상에서는 review_feedback를 반영해 코드를 수정한다.
    """
    iteration = state.get("iteration_count", 0)
    label = "최초 생성" if iteration == 0 else f"피드백 반영 ({iteration}회차 → {iteration + 1}회차)"
    logger.info(f"[Coder] 시작 — {label}")

    # 논문 원문 부분은 Reviewer와 같은 함수로 만든다
    paper_context = paper_source_context(state)

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
            _SKIP_INSTRUCTION
            + "다음 논문을 바탕으로 PyTorch 코드 스켈레톤을 작성하세요.\n\n"
            + (f"### 논문 요약:\n{summary}\n\n" if summary else "")
            + (f"{formula_context}\n\n" if formula_context else "")
            + f"{paper_context}"
        )
    else:
        # 실행 검사가 실패했으면 그 결과(직전 Reviewer 지적 포함)를, 아니면 Reviewer 피드백 + 고정된 smoke_test를 준다
        feedback = state.get("exec_feedback") or state.get("review_feedback", "")
        if not state.get("exec_feedback") and state.get("smoke_test"):
            feedback += "\n\n" + fixed_test_notice(state["smoke_test"])
        user_content = (
            "피드백을 반영해 이전 코드를 수정하세요.\n\n"
            f"### 피드백:\n{feedback}\n\n"
            f"### 이전 코드:\n{state.get('generated_code', '')}\n\n"
            f"### 참고 논문:\n{paper_context}"
        )

    try:
        paper_count = len(state.get("papers", []))
        emit_log("coder", f"논문 {min(paper_count, 3)}편 분석 중...")
        if iteration > 0:
            source = "실행 검사 결과" if state.get("exec_feedback") else "리뷰어 피드백"
            emit_log("coder", f"{source} 반영 ({iteration}회차 수정)")
        emit_log("coder", "PyTorch 코드 생성 중...")
        async with log_elapsed(logger, "external_call", node="coder", external="openai"):
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
            # 이전 코드·피드백까지 합친 전체 입력 + 출력 예산이 한도 안인지 호출 직전에 확인한다
            ensure_request_fits(_MODEL, messages, REASONING_OUTPUT_BUDGET)
            response = await _llm.ainvoke(messages)
            ensure_complete(response, REASONING_OUTPUT_BUDGET)
        skip_reason = _skip_reason(response.content) if iteration == 0 else None
        if skip_reason:
            emit_log("coder", f"구현 대상이 아닌 논문 — 코드 생성 생략: {skip_reason}")
            logger.info(f"[Coder] 구현 대상 아님 — {skip_reason}")
            return {
                "generated_code": "",
                "code_skip_reason": skip_reason,
                "iteration_count": iteration + 1,
                "current_node": "coder",
                "error": None,
            }
        generated_code = _extract_code(response.content)
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
