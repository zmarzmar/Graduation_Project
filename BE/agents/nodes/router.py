from langgraph.graph import END

from agents.exec_check import CODE_ISSUES
from agents.state import AgentState

MAX_ITERATIONS = 3


def after_coder(state: AgentState) -> str:
    """구현 대상이 아니거나 Coder 호출이 실패했으면 끝내고, 아니면 실행 검사로 간다."""
    if state.get("code_skip_reason") or state.get("error"):
        return END
    return "executor"


def after_executor(state: AgentState) -> str:
    """실행 오류면 Coder로(최대 3회, 마지막이면 Reviewer 없이 끝), 그 밖(통과·테스트 형식 위반·실행 확인 못 함)은 Reviewer로."""
    if state.get("exec_status") in CODE_ISSUES:
        return END if state.get("iteration_count", 0) >= MAX_ITERATIONS else "coder"
    return "reviewer"


def route(state: AgentState) -> str:
    """Reviewer 결과에 따라 다음 노드를 결정한다.

    Returns:
        END     — Reviewer 호출·응답 오류(판정과 섞지 않는다), 통과, 또는 최대 반복 횟수(3회)
        "coder" — 낙제이고 반복 횟수가 3회 미만
    """
    if state.get("error") or state.get("review_passed", False) or state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return END
    return "coder"
