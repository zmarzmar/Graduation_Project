import asyncio
import io
import json
import logging
from typing import AsyncGenerator

from pypdf import PdfReader

from agents.graph import agent_graph, analyze_graph
from agents.log_stream import set_log_queue
from core.dependencies import AsyncSessionLocal
from crud import analysis as crud_analysis
from crud import paper as crud_paper
from crud import search_history as crud_search_history
from schemas.paper import PaperResult

logger = logging.getLogger(__name__)

# SSE 이벤트를 전송할 노드 목록 (LangGraph 내부 노드 제외)
_AGENT_NODES = {"planner", "researcher", "trend_analyzer", "analyzer", "coder", "reviewer"}


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
        "paper_summary": "",
        "paper_review": {},
        "key_formulas": [],
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
    elif node_name == "analyzer":
        event["paper_summary"] = updates.get("paper_summary", "")
        event["key_formulas_count"] = len(updates.get("key_formulas", []))
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
    final_result = accumulated.get("final_result") or {}
    final_result.update({
        "papers": accumulated.get("papers", []),
        "paper_summary": accumulated.get("paper_summary", ""),
        "paper_review": accumulated.get("paper_review", {}),
        "key_formulas": accumulated.get("key_formulas", []),
        "generated_code": accumulated.get("generated_code", ""),
        "review_feedback": accumulated.get("review_feedback", ""),
        "mode": mode,
    })

    # DB 저장 — search 모드는 검색 기록만, pdf/trend는 전체 저장
    try:
        await _save_to_db(mode, user_query, accumulated, search_only=(mode == "search"))
    except Exception as e:
        logger.error(f"DB 저장 실패 (무시): {e}")

    yield _sse({"event": "complete", "result": final_result})


async def _save_to_db(
    mode: str,
    user_query: str,
    accumulated: dict,
    search_only: bool = False,
) -> None:
    """에이전트 실행 결과를 DB에 저장한다.

    search_only=True 이면 검색 기록만 저장하고 분석 결과는 저장하지 않는다.
    (search 모드는 Researcher에서 끝나므로 실제 분석 결과가 없음)
    """
    papers: list[dict] = accumulated.get("papers", [])

    async with AsyncSessionLocal() as db:
        # 검색 기록 저장 (항상)
        await crud_search_history.create_search_history(
            db, query=user_query, mode=mode, result_count=len(papers)
        )

        if search_only:
            await db.commit()
            return

        # 논문 저장 (pdf 모드만 — 분석 대상 논문이 명확할 때)
        paper_id: int | None = None
        if papers and mode == "pdf":
            first_paper_dict = papers[0] if isinstance(papers[0], dict) else papers[0].__dict__
            try:
                paper_schema = PaperResult(**first_paper_dict)
                paper_obj = await crud_paper.upsert_paper(db, paper_schema)
                paper_id = paper_obj.id
            except Exception as e:
                logger.warning(f"논문 저장 실패 (무시): {e}")

        # 분석 결과 저장 (pdf/trend 모드)
        await crud_analysis.create_analysis_result(
            db,
            mode=mode,
            query=user_query,
            generated_code=accumulated.get("generated_code", ""),
            review_feedback=accumulated.get("review_feedback", ""),
            review_passed=accumulated.get("review_passed", False),
            iteration_count=accumulated.get("iteration_count", 0),
            paper_id=paper_id,
        )
        await db.commit()


async def stream_analyze(
    paper: dict,
    user_query: str,
) -> AsyncGenerator[str, None]:
    """사용자가 선택한 논문 1편을 Analyzer → Coder → Reviewer로 분석한다."""
    initial_state = _make_initial_state("pdf", user_query)
    initial_state["papers"] = [paper]
    accumulated: dict = dict(initial_state)

    queue: asyncio.Queue = asyncio.Queue()
    set_log_queue(queue)

    async def _run_graph() -> None:
        try:
            async for chunk in analyze_graph.astream(initial_state, stream_mode="updates"):
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
                if name not in started_nodes:
                    started_nodes.add(name)
                    yield _sse({"event": "node_start", "node": name})
                yield _sse({"event": "log", "node": name, "message": data})

            elif kind == "node":
                if name not in started_nodes:
                    started_nodes.add(name)
                    yield _sse({"event": "node_start", "node": name})
                accumulated.update(data)
                yield _sse(_build_node_done_event(name, data))

            elif kind == "error":
                logger.error(f"분석 스트리밍 중 오류: {data}")
                yield _sse({"event": "error", "message": data})
                return

            elif kind == "done":
                break

    finally:
        task.cancel()

    final_result = {
        "papers": [paper],
        "paper_summary": accumulated.get("paper_summary", ""),
        "paper_review": accumulated.get("paper_review", {}),
        "key_formulas": accumulated.get("key_formulas", []),
        "generated_code": accumulated.get("generated_code", ""),
        "review_feedback": accumulated.get("review_feedback", ""),
        "review_passed": accumulated.get("review_passed", False),
        "iteration_count": accumulated.get("iteration_count", 0),
        "mode": "search",
    }

    # DB 저장 — 선택한 논문 + 분석 결과
    try:
        accumulated["papers"] = [paper]
        await _save_analyze_to_db(user_query, paper, accumulated)
    except Exception as e:
        logger.error(f"분석 DB 저장 실패 (무시): {e}")

    yield _sse({"event": "complete", "result": final_result})


async def _save_analyze_to_db(user_query: str, paper: dict, accumulated: dict) -> None:
    """사용자가 선택한 논문 분석 결과를 DB에 저장한다."""
    async with AsyncSessionLocal() as db:
        # 선택한 논문 저장
        paper_id: int | None = None
        try:
            paper_schema = PaperResult(**paper)
            paper_obj = await crud_paper.upsert_paper(db, paper_schema)
            paper_id = paper_obj.id
        except Exception as e:
            logger.warning(f"논문 저장 실패 (무시): {e}")

        # 분석 결과 저장
        await crud_analysis.create_analysis_result(
            db,
            mode="search",
            query=user_query,
            generated_code=accumulated.get("generated_code", ""),
            review_feedback=accumulated.get("review_feedback", ""),
            review_passed=accumulated.get("review_passed", False),
            iteration_count=accumulated.get("iteration_count", 0),
            paper_id=paper_id,
        )
        await db.commit()


def _extract_plan_summary(plan_str: str) -> str:
    """plan JSON 문자열에서 summary 필드만 추출한다."""
    try:
        return json.loads(plan_str).get("summary", "")
    except (json.JSONDecodeError, TypeError):
        return ""
