import asyncio
import io
import json
import logging
import time
from contextlib import suppress
from typing import AsyncGenerator
from urllib.parse import urlparse

import fitz
import httpx

from agents.graph import agent_graph, analyze_graph
from agents.log_stream import set_log_queue
from agents.perf import log_elapsed
from core.dependencies import AsyncSessionLocal
from crud import analysis as crud_analysis
from crud import paper as crud_paper
from crud import search_history as crud_search_history
from schemas.paper import PaperResult

logger = logging.getLogger(__name__)

# SSE 이벤트를 전송할 노드 목록 (LangGraph 내부 노드 제외)
_AGENT_NODES = {"planner", "researcher", "trend_analyzer", "analyzer", "coder", "reviewer"}
_PDF_HEADERS = {"User-Agent": "arxiv-analyst/0.1 (graduation-project; contact@example.com)"}


async def _cancel_task(task: asyncio.Task[None]) -> None:
    """실행 중인 graph task를 취소하고 취소 완료까지 기다린다."""
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def extract_pdf_text(file_bytes: bytes) -> str:
    """PDF 바이트에서 전체 텍스트를 추출한다. (pymupdf 사용)"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


async def download_pdf_text(pdf_url: str) -> str:
    """arXiv PDF URL에서 PDF를 다운로드하고 전체 텍스트를 추출한다."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(pdf_url, headers=_PDF_HEADERS)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    content = response.content
    if "pdf" not in content_type and not content.startswith(b"%PDF"):
        raise ValueError(f"PDF 응답이 아닙니다. content-type={content_type or 'unknown'}")

    text = extract_pdf_text(content).strip()
    if not text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")
    return text


def _normalize_arxiv_id(arxiv_id: str) -> str:
    """arXiv ID를 URL 생성에 쓸 수 있는 형태로 정리한다."""
    return arxiv_id.strip().removeprefix("arXiv:").removesuffix(".pdf")


def _add_pdf_candidate(candidates: list[str], url: str) -> None:
    url = url.strip()
    if url and url not in candidates:
        candidates.append(url)


def _build_pdf_url_candidates(paper: dict) -> list[str]:
    """논문 메타데이터에서 시도 가능한 PDF URL 후보를 만든다."""
    candidates: list[str] = []

    arxiv_id = _normalize_arxiv_id(str(paper.get("arxiv_id") or ""))
    if arxiv_id:
        _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{arxiv_id}")
        _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{arxiv_id}.pdf")

    for key in ("pdf_url", "url"):
        raw_url = str(paper.get(key) or "").strip()
        if not raw_url:
            continue

        parsed = urlparse(raw_url)
        if parsed.netloc.endswith("arxiv.org") and parsed.path.startswith("/abs/"):
            abs_id = _normalize_arxiv_id(parsed.path.removeprefix("/abs/"))
            if abs_id:
                _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{abs_id}")
                _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{abs_id}.pdf")
            continue

        _add_pdf_candidate(candidates, raw_url)

    return candidates


async def _download_first_available_pdf_text(paper: dict) -> tuple[str, str]:
    """PDF 후보 URL을 순서대로 시도하고 성공한 텍스트와 URL을 반환한다."""
    errors: list[str] = []
    candidates = _build_pdf_url_candidates(paper)
    if not candidates:
        raise ValueError("시도 가능한 PDF URL이 없습니다.")

    for pdf_url in candidates:
        try:
            return await download_pdf_text(pdf_url), pdf_url
        except Exception as e:
            errors.append(f"{pdf_url}: {e}")

    raise ValueError("모든 PDF 후보 다운로드 실패: " + " | ".join(errors))


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
    if (elapsed_ms := updates.get("elapsed_ms")) is not None:
        event["elapsed_ms"] = elapsed_ms

    return event


async def stream_agent(
    mode: str,
    user_query: str,
    pdf_text: str = "",
    user_id: int | None = None,
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
    node_started_at: dict[str, float] = {}

    try:
        while True:
            item = await queue.get()
            kind, name, data = item

            if kind == "log":
                # 노드 첫 로그 도착 시 node_start 이벤트 선행 전송
                if name not in started_nodes:
                    started_nodes.add(name)
                    node_started_at[name] = time.perf_counter()
                    yield _sse({"event": "node_start", "node": name})
                yield _sse({"event": "log", "node": name, "message": data})

            elif kind == "node":
                # 노드 완료 — node_start 미전송 상태면 여기서 전송
                if name not in started_nodes:
                    started_nodes.add(name)
                    node_started_at[name] = time.perf_counter()
                    yield _sse({"event": "node_start", "node": name})
                accumulated.update(data)
                if started_at := node_started_at.get(name):
                    data = {**data, "elapsed_ms": int((time.perf_counter() - started_at) * 1000)}
                yield _sse(_build_node_done_event(name, data))
                node_started_at.pop(name, None)
                started_nodes.discard(name)

            elif kind == "error":
                logger.error(f"에이전트 스트리밍 중 오류: {data}")
                yield _sse({"event": "error", "message": data})
                return

            elif kind == "done":
                break

    finally:
        await _cancel_task(task)

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

    # DB 저장 — search/trend 모드는 검색 기록만, pdf는 전체 저장
    try:
        await _save_to_db(mode, user_query, accumulated, search_only=(mode in ("search", "trend")), user_id=user_id)
    except Exception as e:
        logger.error(f"DB 저장 실패 (무시): {e}")

    yield _sse({"event": "complete", "result": final_result})


async def _save_to_db(
    mode: str,
    user_query: str,
    accumulated: dict,
    search_only: bool = False,
    user_id: int | None = None,
) -> None:
    """에이전트 실행 결과를 DB에 저장한다.

    search_only=True 이면 검색 기록만 저장하고 분석 결과는 저장하지 않는다.
    (search 모드는 Researcher에서 끝나므로 실제 분석 결과가 없음)
    """
    papers: list[dict] = accumulated.get("papers", [])

    async with AsyncSessionLocal() as db:
        # 검색 기록 저장 (항상)
        await crud_search_history.create_search_history(
            db, query=user_query, mode=mode, result_count=len(papers), papers=papers, user_id=user_id
        )

        if search_only:
            await db.commit()
            return

        # PDF 모드는 analyzer가 업로드된 PDF 본문을 기반으로 분석하지만,
        # researcher가 수집한 papers[0]은 키워드 검색 1위 결과(다른 논문일 수 있음)라
        # paper_id를 연결하지 않는다. 업로드된 논문의 식별은 user_query(파일명)와
        # paper_summary로 충분하고, 향후 PDF 본문에서 arxiv_id를 추출할 수 있게 되면
        # 그때 정확 매칭을 붙인다.
        await crud_analysis.create_analysis_result(
            db,
            mode=mode,
            query=user_query,
            generated_code=accumulated.get("generated_code", ""),
            review_feedback=accumulated.get("review_feedback", ""),
            review_passed=accumulated.get("review_passed", False),
            iteration_count=accumulated.get("iteration_count", 0),
            paper_id=None,
            paper_summary=accumulated.get("paper_summary", ""),
            paper_review=accumulated.get("paper_review", {}),
            key_formulas=accumulated.get("key_formulas", []),
            user_id=user_id,
        )
        await db.commit()


async def stream_analyze(
    paper: dict,
    user_query: str,
    allow_abstract_fallback: bool = False,
    user_id: int | None = None,
) -> AsyncGenerator[str, None]:
    """사용자가 선택한 논문 1편을 Analyzer → Coder → Reviewer로 분석한다.
    pdf_url이 있으면 arXiv PDF 전체 텍스트를 다운로드해서 분석에 사용한다.
    """
    pdf_text = ""
    analysis_source = "pdf"

    try:
        yield _sse({"event": "log", "node": "analyzer", "message": "arXiv PDF 다운로드 중..."})
        async with log_elapsed(logger, "external_call", node="analyze", external="pdf_download"):
            pdf_text, used_pdf_url = await _download_first_available_pdf_text(paper)
        yield _sse({
            "event": "log",
            "node": "analyzer",
            "message": f"PDF 전체 텍스트 추출 완료 ({len(pdf_text)}자)",
            "pdf_url": used_pdf_url,
        })
    except Exception as e:
        logger.warning(f"PDF 다운로드 실패: {e}")
        if not allow_abstract_fallback:
            yield _sse({
                "event": "pdf_fallback_required",
                "node": "analyzer",
                "message": "PDF를 다운로드하지 못했습니다. 초록만으로 분석을 진행할까요?",
                "reason": "download_failed",
            })
            return

        if not paper.get("abstract"):
            yield _sse({"event": "error", "message": "PDF를 다운로드하지 못했고 초록도 없어 분석을 진행할 수 없습니다."})
            return

        analysis_source = "abstract"
        yield _sse({"event": "log", "node": "analyzer", "message": "사용자 동의에 따라 초록으로 분석 진행"})

    initial_state = _make_initial_state("analyze", user_query, pdf_text=pdf_text)
    initial_state["papers"] = [paper]
    initial_state["analysis_source"] = analysis_source
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
    node_started_at: dict[str, float] = {}

    try:
        while True:
            item = await queue.get()
            kind, name, data = item

            if kind == "log":
                if name not in started_nodes:
                    started_nodes.add(name)
                    node_started_at[name] = time.perf_counter()
                    yield _sse({"event": "node_start", "node": name})
                yield _sse({"event": "log", "node": name, "message": data})

            elif kind == "node":
                if name not in started_nodes:
                    started_nodes.add(name)
                    node_started_at[name] = time.perf_counter()
                    yield _sse({"event": "node_start", "node": name})
                accumulated.update(data)
                if started_at := node_started_at.get(name):
                    data = {**data, "elapsed_ms": int((time.perf_counter() - started_at) * 1000)}
                yield _sse(_build_node_done_event(name, data))
                node_started_at.pop(name, None)
                started_nodes.discard(name)

            elif kind == "error":
                logger.error(f"분석 스트리밍 중 오류: {data}")
                yield _sse({"event": "error", "message": data})
                return

            elif kind == "done":
                break

    finally:
        await _cancel_task(task)

    final_result = {
        "papers": [paper],
        "paper_summary": accumulated.get("paper_summary", ""),
        "paper_review": accumulated.get("paper_review", {}),
        "key_formulas": accumulated.get("key_formulas", []),
        "generated_code": accumulated.get("generated_code", ""),
        "review_feedback": accumulated.get("review_feedback", ""),
        "review_passed": accumulated.get("review_passed", False),
        "iteration_count": accumulated.get("iteration_count", 0),
        "mode": "analyze",
        "analysis_source": analysis_source,
    }

    # DB 저장 — 선택한 논문 + 분석 결과
    try:
        accumulated["papers"] = [paper]
        await _save_analyze_to_db(user_query, paper, accumulated, user_id=user_id)
    except Exception as e:
        logger.error(f"분석 DB 저장 실패 (무시): {e}")

    yield _sse({"event": "complete", "result": final_result})


async def _save_analyze_to_db(user_query: str, paper: dict, accumulated: dict, user_id: int | None = None) -> None:
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
            mode="analyze",
            query=user_query,
            generated_code=accumulated.get("generated_code", ""),
            review_feedback=accumulated.get("review_feedback", ""),
            review_passed=accumulated.get("review_passed", False),
            iteration_count=accumulated.get("iteration_count", 0),
            paper_id=paper_id,
            paper_summary=accumulated.get("paper_summary", ""),
            paper_review=accumulated.get("paper_review", {}),
            key_formulas=accumulated.get("key_formulas", []),
            user_id=user_id,
        )
        await db.commit()


def _extract_plan_summary(plan_str: str) -> str:
    """plan JSON 문자열에서 summary 필드만 추출한다."""
    try:
        return json.loads(plan_str).get("summary", "")
    except (json.JSONDecodeError, TypeError):
        return ""
