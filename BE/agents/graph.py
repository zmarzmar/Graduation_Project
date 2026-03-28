from langgraph.graph import END, START, StateGraph

from agents.nodes.analyzer import analyzer_node
from agents.nodes.coder import coder_node
from agents.nodes.planner import planner_node
from agents.nodes.researcher import researcher_node
from agents.nodes.reviewer import reviewer_node
from agents.nodes.router import route
from agents.nodes.trend_analyzer import trend_analyzer_node
from agents.state import AgentState


def _after_researcher(state: AgentState) -> str:
    """Researcher 완료 후 분기를 결정한다.
    - trend → TrendAnalyzer
    - pdf   → Analyzer (자동 분석)
    - search → END (사용자가 논문 선택 후 /agent/analyze로 분석 시작)
    """
    mode = state.get("mode")
    if mode == "trend":
        return "trend_analyzer"
    if mode == "pdf":
        return "analyzer"
    return END


def build_analyze_graph() -> StateGraph:
    """Analyzer → Coder → Reviewer 분석 전용 그래프.
    사용자가 논문을 선택한 뒤 /agent/analyze 엔드포인트에서 사용한다.
    """
    graph = StateGraph(AgentState)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_conditional_edges("reviewer", route)

    return graph.compile()


def build_graph() -> StateGraph:
    """LangGraph 에이전트 파이프라인 그래프를 생성하고 컴파일한다.

    그래프 구조:
        START → planner → researcher → (trend: trend_analyzer → END)
                                               ↓ (pdf/search)
                                           analyzer → coder → reviewer → router → (통과/초과: END | 낙제: coder)
    """
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("trend_analyzer", trend_analyzer_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)

    # 엣지 연결
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")

    # Researcher → TrendAnalyzer (trend 모드) 또는 Analyzer (pdf/search 모드)
    graph.add_conditional_edges("researcher", _after_researcher)

    graph.add_edge("trend_analyzer", END)
    graph.add_edge("analyzer", "coder")
    graph.add_edge("coder", "reviewer")

    # Reviewer → Coder (낙제, 3회 미만) 또는 END (통과 또는 3회 초과)
    graph.add_conditional_edges("reviewer", route)

    return graph.compile()


# 싱글톤 인스턴스 — 모듈 임포트 시 한 번만 컴파일
agent_graph = build_graph()
analyze_graph = build_analyze_graph()
