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
from agents.token_budget import TokenizerUnavailableError, count_tokens
from core.config import settings
from core.dependencies import AsyncSessionLocal
from crud import analysis as crud_analysis
from crud import paper as crud_paper
from crud import search_history as crud_search_history
from schemas.paper import PaperResult

logger = logging.getLogger(__name__)

# SSE 이벤트를 전송할 노드 목록 (LangGraph 내부 노드 제외)
_AGENT_NODES = {"planner", "researcher", "trend_analyzer", "analyzer", "coder", "reviewer"}
_PDF_HEADERS = {"User-Agent": "arxiv-analyst/0.1 (graduation-project; contact@example.com)"}
# ponytail: 고정된 허용 호스트라 DNS → 내부 IP 검사는 생략. 임의 도메인을 허용하게 되면
# 해석된 IP의 사설/루프백/링크로컬 대역 차단을 추가해야 한다.
_ALLOWED_PDF_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
_MAX_PDF_REDIRECTS = 3


async def _cancel_task(task: asyncio.Task[None]) -> None:
    """실행 중인 graph task를 취소하고 취소 완료까지 기다린다."""
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class PaperTooLongError(ValueError):
    """논문 본문이 모델 입력 한도를 넘는다. 자르지 않고 사용자에게 알린다."""


def join_pages(pages: list[str]) -> str:
    """페이지 목록을 노드에 전달할 전문 텍스트로 합친다."""
    return "\n".join(pages).strip()


def extract_pdf_pages(file_bytes: bytes) -> list[str]:
    """PDF 바이트에서 페이지별 텍스트를 추출한다. (pymupdf 사용)

    빈 페이지도 그대로 둔다 — 인덱스 + 1이 원본 PDF 페이지 번호와 일치해야 출처 표시가 맞는다.
    업로드·다운로드 양쪽 경로가 모두 이 함수를 지나므로 본문 검증도 여기서 한다.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    text = join_pages(pages)

    # 스캔본 등 텍스트 레이어가 없는 PDF — 빈 본문으로 분석을 시작하지 않도록 여기서 막는다.
    if not text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. (스캔 이미지 PDF는 지원하지 않습니다)")

    tokens = count_tokens(text)
    if tokens > settings.max_paper_tokens:
        raise PaperTooLongError(
            f"논문이 너무 깁니다 ({len(pages)}쪽, {tokens:,} 토큰 — 한도 {settings.max_paper_tokens:,} 토큰). "
            "본문을 잘라서 분석하면 Methods·부록이 빠질 수 있어 전문 분석을 진행하지 않습니다."
        )
    return pages


def _validate_pdf_url(url: str) -> str:
    """서버가 직접 방문해도 되는 PDF URL인지 검증한다. (SSRF 방지)

    의도적인 기능 축소: arXiv 외 출처(출판사·대학 저장소 등)의 PDF는 받지 않는다.
    해당 논문은 호출부의 초록 기반 분석 폴백으로 넘어간다.
    """
    # 실제 요청을 보내는 httpx와 같은 파서를 써서 파서 간 해석 차이를 없앤다.
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as e:
        raise ValueError(f"잘못된 PDF URL: {e}") from e

    if parsed.scheme != "https":
        raise ValueError(f"허용되지 않은 스킴: {parsed.scheme or '(없음)'}")
    if parsed.userinfo:
        raise ValueError("userinfo가 포함된 URL은 허용되지 않습니다.")
    # 부분 일치가 아닌 정확 일치 — evilarxiv.org, arxiv.org.evil.com 차단
    if parsed.host not in _ALLOWED_PDF_HOSTS:
        raise ValueError(f"허용되지 않은 PDF 출처: {parsed.host or '(없음)'}")
    if parsed.port not in (None, 443):
        raise ValueError(f"허용되지 않은 포트: {parsed.port}")
    return str(parsed)


async def download_pdf_pages(pdf_url: str) -> list[str]:
    """arXiv PDF URL에서 PDF를 다운로드하고 페이지별 텍스트를 추출한다."""
    url = _validate_pdf_url(pdf_url)
    # 자동 리다이렉트를 끄고 매 홉마다 목적지를 재검증한다.
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        for _ in range(_MAX_PDF_REDIRECTS + 1):
            response = await client.get(url, headers=_PDF_HEADERS)
            if not response.is_redirect:
                break
            # 상대 경로 Location은 현재 URL 기준으로 풀어서 검증한다.
            url = _validate_pdf_url(str(response.url.join(response.headers["location"])))
        else:
            raise ValueError(f"리다이렉트 {_MAX_PDF_REDIRECTS}회 초과")
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    content = response.content
    if "pdf" not in content_type and not content.startswith(b"%PDF"):
        raise ValueError(f"PDF 응답이 아닙니다. content-type={content_type or 'unknown'}")

    return extract_pdf_pages(content)


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
        if parsed.hostname in _ALLOWED_PDF_HOSTS and parsed.path.startswith("/abs/"):
            abs_id = _normalize_arxiv_id(parsed.path.removeprefix("/abs/"))
            if abs_id:
                _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{abs_id}")
                _add_pdf_candidate(candidates, f"https://arxiv.org/pdf/{abs_id}.pdf")
            continue

        _add_pdf_candidate(candidates, raw_url)

    return candidates


async def _download_first_available_pdf_pages(paper: dict) -> tuple[list[str], str]:
    """PDF 후보 URL을 순서대로 시도하고 성공한 페이지 목록과 URL을 반환한다."""
    errors: list[str] = []
    candidates = _build_pdf_url_candidates(paper)
    if not candidates:
        raise ValueError("시도 가능한 PDF URL이 없습니다.")

    for pdf_url in candidates:
        try:
            return await download_pdf_pages(pdf_url), pdf_url
        except (PaperTooLongError, TokenizerUnavailableError):
            # 다운로드는 성공했다 — 다른 후보를 시도해도 결과가 같으므로 그대로 알린다.
            raise
        except Exception as e:
            errors.append(f"{pdf_url}: {e}")

    raise ValueError("모든 PDF 후보 다운로드 실패: " + " | ".join(errors))


def _make_initial_state(
    mode: str,
    user_query: str,
    pdf_pages: list[str] | None = None,
) -> dict:
    """에이전트 초기 상태를 생성한다. pdf_text는 pdf_pages에서 파생한다 (PDF는 한 번만 파싱)."""
    pdf_pages = pdf_pages or []
    return {
        "mode": mode,
        "user_query": user_query,
        "pdf_pages": pdf_pages,
        "pdf_text": join_pages(pdf_pages),
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
    pdf_pages: list[str] | None = None,
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
    initial_state = _make_initial_state(mode, user_query, pdf_pages)
    accumulated: dict = dict(initial_state)

    # 로그 메시지와 노드 완료 청크를 하나의 채널로 합친다
    queue: asyncio.Queue = asyncio.Queue()
    set_log_queue(queue)

    # PDF 모드는 본문이 이미 있으므로 Planner·Researcher(관련 논문 검색)를 거치지 않는다.
    graph = analyze_graph if mode == "pdf" else agent_graph

    async def _run_graph() -> None:
        """그래프를 실행하며 완료 청크를 queue에 넣는다."""
        try:
            async for chunk in graph.astream(initial_state, stream_mode="updates"):
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
    if mode == "pdf":
        # 분석 대상은 업로드한 파일 — papers(검색 결과)와 구분해서 전달한다.
        final_result["uploaded_filename"] = user_query

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
        # 검색 기록 저장 — PDF 모드는 검색을 하지 않으므로 분석 이력만 남긴다.
        if mode != "pdf":
            await crud_search_history.create_search_history(
                db, query=user_query, mode=mode, result_count=len(papers), papers=papers, user_id=user_id
            )

        if search_only:
            await db.commit()
            return

        # PDF 모드는 업로드된 본문을 분석하므로 연결할 Paper 행이 없다(paper_id=None).
        # 업로드된 논문의 식별은 user_query(파일명)와 paper_summary로 충분하고,
        # 향후 PDF 본문에서 arxiv_id를 추출할 수 있게 되면 그때 정확 매칭을 붙인다.
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
    pdf_pages: list[str] = []
    analysis_source = "pdf"

    try:
        yield _sse({"event": "log", "node": "analyzer", "message": "arXiv PDF 다운로드 중..."})
        async with log_elapsed(logger, "external_call", node="analyze", external="pdf_download"):
            pdf_pages, used_pdf_url = await _download_first_available_pdf_pages(paper)
        yield _sse({
            "event": "log",
            "node": "analyzer",
            "message": f"PDF 전체 텍스트 추출 완료 ({len(pdf_pages)}쪽, {len(join_pages(pdf_pages))}자)",
            "pdf_url": used_pdf_url,
        })
    except TokenizerUnavailableError as e:
        # 길이를 검증할 수 없으면 초록 폴백을 권하지 않고 그대로 알린다 — 다운로드 실패가 아니다.
        logger.error(f"토크나이저 사용 불가: {e}")
        yield _sse({"event": "error", "message": str(e)})
        return
    except Exception as e:
        # 너무 긴 논문은 다운로드 실패와 구분해서 알린다 — 잘라서 분석하지 않는다.
        too_long = isinstance(e, PaperTooLongError)
        problem = str(e) if too_long else "PDF를 다운로드하지 못했습니다."
        logger.warning(f"PDF 전문 사용 불가: {e}")
        if not allow_abstract_fallback:
            yield _sse({
                "event": "pdf_fallback_required",
                "node": "analyzer",
                "message": f"{problem} 초록만으로 분석을 진행할까요?",
                "reason": "paper_too_long" if too_long else "download_failed",
            })
            return

        if not paper.get("abstract"):
            yield _sse({"event": "error", "message": f"{problem} 초록도 없어 분석을 진행할 수 없습니다."})
            return

        analysis_source = "abstract"
        yield _sse({"event": "log", "node": "analyzer", "message": "사용자 동의에 따라 초록으로 분석 진행"})

    initial_state = _make_initial_state("analyze", user_query, pdf_pages=pdf_pages)
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
