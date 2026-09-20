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
    - search → END (사용자가 논문 선택 후 /agent/analyze로 분석 시작)
    """
    return "trend_analyzer" if state.get("mode") == "trend" else END


def build_analyze_graph() -> StateGraph:
    """Analyzer → Coder → Reviewer 분석 전용 그래프.
    분석할 본문이 이미 정해진 경우에 사용한다 — 검색 후 논문 선택(/agent/analyze), PDF 업로드(/agent/pdf).
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
    """논문 검색용 그래프를 생성하고 컴파일한다. (search / trend 모드)

    그래프 구조:
        START → planner → researcher → (trend: trend_analyzer → END | search: END)

    코드 생성·자기수정 루프는 build_analyze_graph()가 담당한다.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("trend_analyzer", trend_analyzer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_conditional_edges("researcher", _after_researcher)
    graph.add_edge("trend_analyzer", END)

    return graph.compile()


# 싱글톤 인스턴스 — 모듈 임포트 시 한 번만 컴파일
agent_graph = build_graph()
analyze_graph = build_analyze_graph()
