"""Researcher 노드 — 선택 기능인 S2 인용 수 보완이 실패해도 수집한 논문을 버리지 않는다."""

import asyncio
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import httpx

from agents.nodes import researcher
from schemas.paper import PaperResult
from services import semantic_scholar_service


def _arxiv_paper() -> PaperResult:
    return PaperResult(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=["A. Vaswani"],
        abstract="The dominant sequence transduction models ...",
        url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762",
        published_at=datetime(2017, 6, 12, tzinfo=timezone.utc),
        categories=["cs.CL"],
    )


class EnrichFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_keeps_arxiv_papers_when_enrich_fails(self) -> None:
        # 2026-11-05 운영: S2 검색 403, OpenAlex 429, arXiv 성공, S2 보완 403 → 0편이 되던 상황
        state = {
            "mode": "search",
            "plan": json.dumps({"search_keywords": ["attention"]}),
            "user_query": "attention",
        }
        with (
            patch.object(researcher.semantic_scholar_service, "search_papers", AsyncMock(side_effect=HTTPException(502, "403"))),
            patch.object(researcher.openalex_service, "search_papers", AsyncMock(side_effect=HTTPException(429, "429"))),
            patch.object(researcher.arxiv_service, "search_papers", AsyncMock(return_value=[_arxiv_paper()])),
            patch.object(
                researcher.semantic_scholar_service,
                "enrich_papers",
                AsyncMock(side_effect=HTTPException(502, "Semantic Scholar API 오류: 403")),
            ),
        ):
            result = await researcher.researcher_node(state)

        self.assertIsNone(result["error"])
        self.assertEqual([p["title"] for p in result["papers"]], ["Attention Is All You Need"])
        self.assertIsNone(result["papers"][0]["citation_count"])

    async def test_trend_keeps_arxiv_papers_without_success_log_when_enrich_fails(self) -> None:
        state = {
            "mode": "trend",
            "plan": json.dumps({"search_keywords": ["attention"]}),
            "user_query": "attention",
        }
        logs: list[str] = []
        with (
            patch.object(researcher, "_fetch_hf_trending", AsyncMock(return_value=[])),
            patch.object(researcher, "emit_log", lambda node, msg: logs.append(msg)),
            patch.object(researcher.semantic_scholar_service, "search_papers", AsyncMock(side_effect=HTTPException(502, "403"))),
            patch.object(researcher.arxiv_service, "search_papers", AsyncMock(return_value=[_arxiv_paper()])),
            patch.object(
                researcher.semantic_scholar_service,
                "enrich_papers",
                AsyncMock(side_effect=HTTPException(502, "Semantic Scholar API 오류: 403")),
            ),
        ):
            result = await researcher.researcher_node(state)

        self.assertEqual([p["title"] for p in result["papers"]], ["Attention Is All You Need"])
        self.assertIn("인용 수 보완 실패 — 인용 수 없이 계속 진행", logs)
        self.assertFalse([m for m in logs if m.startswith("인용 수 보완 완료")])


class SearchQueryAndOrderTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_uses_first_keyword_and_sorts_by_citations(self) -> None:
        # 키워드를 이어 붙이지 않고 첫 키워드로만 검색, 결과는 인용 수 순 (없는 것은 뒤로)
        state = {
            "mode": "search",
            "plan": json.dumps({"search_keywords": ["attention mechanism", "transformer models", "deep learning"]}),
            "user_query": "attention mechanism",
        }
        s2 = AsyncMock(return_value=[{"title": "low", "citation_count": 3}, {"title": "none", "citation_count": None}])
        oa = AsyncMock(return_value=[{"title": "high", "citation_count": 5000}] + [{"title": f"oa{i}", "citation_count": 1} for i in range(8)])
        with (
            patch.object(researcher.semantic_scholar_service, "search_papers", s2),
            patch.object(researcher.openalex_service, "search_papers", oa),
            patch.object(researcher.arxiv_service, "search_papers", AsyncMock(return_value=[])),
        ):
            result = await researcher.researcher_node(state)

        self.assertEqual(s2.await_args.args[0], "attention mechanism")
        self.assertEqual(oa.await_args.args[0], "attention mechanism")
        titles = [p["title"] for p in result["papers"]]
        self.assertEqual(len(titles), 10)
        self.assertEqual(titles[:2], ["high", "low"])
        self.assertNotIn("none", titles)  # 11편 중 인용 수 없는 것이 잘린다

    async def test_search_skips_untranslated_korean_keyword(self) -> None:
        state = {
            "mode": "search",
            "plan": json.dumps({"search_keywords": ["어텐션 메커니즘", "attention mechanism", "transformer model"]}),
            "user_query": "어텐션 메커니즘",
        }
        s2 = AsyncMock(return_value=[])
        with (
            patch.object(researcher.semantic_scholar_service, "search_papers", s2),
            patch.object(researcher.openalex_service, "search_papers", AsyncMock(return_value=[])),
            patch.object(researcher.arxiv_service, "search_papers", AsyncMock(return_value=[])),
        ):
            await researcher.researcher_node(state)

        self.assertEqual(s2.await_args.args[0], "attention mechanism")

    async def test_search_dedups_same_title_with_and_without_arxiv_id(self) -> None:
        state = {"mode": "search", "plan": json.dumps({"search_keywords": ["lora"]}), "user_query": "lora"}
        s2 = AsyncMock(return_value=[{"title": "LoftQ: LoRA-Fine-Tuning-Aware Quantization", "arxiv_id": "2310.08659", "citation_count": 267}])
        oa = AsyncMock(return_value=[{"title": "LoftQ: LoRA-Fine-Tuning-Aware Quantization ", "arxiv_id": "", "citation_count": 77}])
        with (
            patch.object(researcher.semantic_scholar_service, "search_papers", s2),
            patch.object(researcher.openalex_service, "search_papers", oa),
            patch.object(researcher.arxiv_service, "search_papers", AsyncMock(return_value=[])),
        ):
            result = await researcher.researcher_node(state)

        self.assertEqual([p["citation_count"] for p in result["papers"]], [267])


class S2RateLimitTest(unittest.IsolatedAsyncioTestCase):
    async def test_send_retries_429_and_keeps_interval(self) -> None:
        statuses = iter([429, 429, 200])
        sleeps: list[float] = []
        clock = [1000.0]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(round(seconds, 2))
            clock[0] += seconds

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(next(statuses))

        with (
            patch.object(semantic_scholar_service.asyncio, "sleep", fake_sleep),
            patch.object(semantic_scholar_service.time, "monotonic", lambda: clock[0]),
            patch.object(semantic_scholar_service, "_last_request_at", 0.0),
        ):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                response = await semantic_scholar_service._send(client, "GET", "https://api.semanticscholar.org/x")

        self.assertEqual(response.status_code, 200)
        # 429 뒤 1초·2초 백오프, 그 사이 다음 요청이 최소 간격(1.1초)을 지킨다
        self.assertEqual(sleeps, [1.0, 0.1, 2.0])

    async def test_send_gives_up_after_retries(self) -> None:
        async def fake_sleep(seconds: float) -> None:
            pass

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429)

        with patch.object(semantic_scholar_service.asyncio, "sleep", fake_sleep):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                response = await semantic_scholar_service._send(client, "GET", "https://api.semanticscholar.org/x")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(calls), 3)  # 처음 1번 + 재시도 2번

    async def test_search_and_batch_share_one_interval_when_concurrent(self) -> None:
        # 검색과 batch를 동시에 불러도 실제 전송은 1.1초 이상 떨어진다 (두 엔드포인트 합산 제한)
        clock = [1000.0]
        sent_at: list[float] = []
        real_sleep = asyncio.sleep  # asyncio.sleep을 바꿔치기하므로 양보용 원본을 잡아 둔다

        async def fake_sleep(seconds: float) -> None:
            clock[0] += seconds
            await real_sleep(0)

        def handler(request: httpx.Request) -> httpx.Response:
            sent_at.append(clock[0])
            if request.method == "POST":
                return httpx.Response(200, json=[None])
            return httpx.Response(200, json={"data": []})

        real_client = httpx.AsyncClient
        with (
            patch.object(semantic_scholar_service.asyncio, "sleep", fake_sleep),
            patch.object(semantic_scholar_service.time, "monotonic", lambda: clock[0]),
            patch.object(semantic_scholar_service, "_last_request_at", 0.0),
            patch.object(
                semantic_scholar_service.httpx,
                "AsyncClient",
                lambda **kw: real_client(transport=httpx.MockTransport(handler)),
            ),
        ):
            await asyncio.gather(
                semantic_scholar_service.search_papers("q"),
                semantic_scholar_service.enrich_papers(["1706.03762"]),
                semantic_scholar_service.search_papers("q2"),
            )

        self.assertEqual(len(sent_at), 3)
        gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
        self.assertTrue(all(g >= 1.1 - 1e-9 for g in gaps), gaps)


if __name__ == "__main__":
    unittest.main()
