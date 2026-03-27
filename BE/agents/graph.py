from langgraph.graph import END, START, StateGraph

from agents.nodes.coder import coder_node
from agents.nodes.planner import planner_node
from agents.nodes.researcher import researcher_node
from agents.nodes.reviewer import reviewer_node
from agents.nodes.router import route
from agents.nodes.trend_analyzer import trend_analyzer_node
from agents.state import AgentState


def _after_researcher(state: AgentState) -> str:
    """Researcher 완료 후 분기를 결정한다.
    trend 모드는 TrendAnalyzer로, 나머지는 Coder로 전송한다.
    """
    if state.get("mode") == "trend":
        return "trend_analyzer"
    return "coder"


def build_graph() -> StateGraph:
    """LangGraph 에이전트 파이프라인 그래프를 생성하고 컴파일한다.

    그래프 구조:
        START → planner → researcher → (trend: trend_analyzer → END)
                                               ↓ (pdf/search)
                                            coder → reviewer → router → (통과/초과: END | 낙제: coder)
    """
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("trend_analyzer", trend_analyzer_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)

    # 엣지 연결
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")

    # Researcher → TrendAnalyzer (trend 모드) 또는 Coder (pdf/search 모드)
    graph.add_conditional_edges("researcher", _after_researcher)

    graph.add_edge("trend_analyzer", END)
    graph.add_edge("coder", "reviewer")

    # Reviewer → Coder (낙제, 3회 미만) 또는 END (통과 또는 3회 초과)
    graph.add_conditional_edges("reviewer", route)

    return graph.compile()


# 싱글톤 인스턴스 — 모듈 임포트 시 한 번만 컴파일
agent_graph = build_graph()
