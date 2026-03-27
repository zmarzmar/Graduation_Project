from langgraph.graph import END

from agents.state import AgentState


def route(state: AgentState) -> str:
    """Reviewer 결과에 따라 다음 노드를 결정한다.

    Returns:
        "coder" — 낙제이고 반복 횟수가 3회 미만
        END     — 통과하거나 최대 반복 횟수(3회) 초과
    """
    if state.get("review_passed", False) or state.get("iteration_count", 0) >= 3:
        return END
    return "coder"
