import asyncio
import io
import json
import logging
from typing import AsyncGenerator

from pypdf import PdfReader

from agents.graph import agent_graph
from agents.log_stream import set_log_queue

logger = logging.getLogger(__name__)

# SSE 이벤트를 전송할 노드 목록 (LangGraph 내부 노드 제외)
_AGENT_NODES = {"planner", "researcher", "trend_analyzer", "coder", "reviewer"}


def extract_pdf_text(file_bytes: bytes) -> str:
    """PDF 바이트에서 전체 텍스트를 추출한다."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _make_initial_state(
    mode: str,
    user_query: str,
    pdf_text: str = "",
) -> dict:
    """에이전트 초기 상태를 생성한다."""
    return {
        "mode": mode,
        "user_query": user_query,
        "pdf_text": pdf_text,
        "plan": "",
        "papers": [],
        "generated_code": "",
        "review_feedback": "",
        "review_passed": False,
        "iteration_count": 0,
        "trend_analysis": {},
        "final_result": {},
        "current_node": "",
        "error": None,
    }


def _sse(payload: dict) -> str:
    """dict를 SSE 형식 문자열로 변환한다."""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _build_node_done_event(node_name: str, updates: dict) -> dict:
    """노드 완료 SSE 이벤트 payload를 구성한다."""
    event: dict = {"event": "node_done", "node": node_name}

    if node_name == "planner":
        event["plan_summary"] = _extract_plan_summary(updates.get("plan", ""))
    elif node_name == "researcher":
        event["papers_count"] = len(updates.get("papers", []))
    elif node_name == "trend_analyzer":
        analysis = updates.get("trend_analysis", {})
        event["summaries_count"] = len(analysis.get("paper_summaries", []))
        event["keywords_count"] = len(analysis.get("trending_keywords", []))
    elif node_name == "coder":
        event["iteration"] = updates.get("iteration_count", 1)
    elif node_name == "reviewer":
        event["review_passed"] = updates.get("review_passed", False)
        event["review_feedback"] = updates.get("review_feedback", "")

    if error := updates.get("error"):
        event["error"] = error

    return event


async def stream_agent(
    mode: str,
    user_query: str,
    pdf_text: str = "",
) -> AsyncGenerator[str, None]:
    """LangGraph 그래프를 실행하고 노드 로그 및 완료 이벤트를 실시간 SSE로 스트리밍한다.

    SSE 이벤트 형식:
        로그        : {"event": "log",       "node": "researcher", "message": "Semantic Scholar 검색 중..."}
        노드 시작   : {"event": "node_start", "node": "planner"}
        노드 완료   : {"event": "node_done",  "node": "planner", ...}
        파이프라인 완료: {"event": "complete", "result": {...}}
        오류        : {"event": "error",     "message": "..."}
    """
    initial_state = _make_initial_state(mode, user_query, pdf_text)
    accumulated: dict = dict(initial_state)

    # 로그 메시지와 노드 완료 청크를 하나의 채널로 합친다
    queue: asyncio.Queue = asyncio.Queue()
    set_log_queue(queue)

    async def _run_graph() -> None:
        """그래프를 실행하며 완료 청크를 queue에 넣는다."""
        try:
            async for chunk in agent_graph.astream(initial_state, stream_mode="updates"):
                for node_name, updates in chunk.items():
                    if node_name in _AGENT_NODES:
                        await queue.put(("node", node_name, updates))
        except Exception as e:
            await queue.put(("error", None, str(e)))
        finally:
            await queue.put(("done", None, None))

    task = asyncio.create_task(_run_graph())
    started_nodes: set[str] = set()

    try:
        while True:
            item = await queue.get()
            kind, name, data = item

            if kind == "log":
                # 노드 첫 로그 도착 시 node_start 이벤트 선행 전송
                if name not in started_nodes:
                    started_nodes.add(name)
                    yield _sse({"event": "node_start", "node": name})
                yield _sse({"event": "log", "node": name, "message": data})

            elif kind == "node":
                # 노드 완료 — node_start 미전송 상태면 여기서 전송
                if name not in started_nodes:
                    started_nodes.add(name)
                    yield _sse({"event": "node_start", "node": name})
                accumulated.update(data)
                yield _sse(_build_node_done_event(name, data))

            elif kind == "error":
                logger.error(f"에이전트 스트리밍 중 오류: {data}")
                yield _sse({"event": "error", "message": data})
                return

            elif kind == "done":
                break

    finally:
        task.cancel()

    # 최종 결과 전송
    final_result = accumulated.get("final_result") or {
        "papers": accumulated.get("papers", []),
        "mode": mode,
    }
    yield _sse({"event": "complete", "result": final_result})


def _extract_plan_summary(plan_str: str) -> str:
    """plan JSON 문자열에서 summary 필드만 추출한다."""
    try:
        return json.loads(plan_str).get("summary", "")
    except (json.JSONDecodeError, TypeError):
        return ""
