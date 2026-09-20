"""PDF 모드 회귀 테스트 — 검색 없이 업로드 본문만 분석하는지 확인한다.

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

import fitz

from agents.graph import agent_graph, analyze_graph
from services import agent_service
from services.agent_service import extract_pdf_text, stream_agent


class _FakeGraph:
    """astream 호출 여부를 기록하고 준비된 노드 업데이트를 흘려보낸다."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.called = False

    async def astream(self, _state: dict, stream_mode: str = "updates"):
        self.called = True
        for chunk in self.chunks:
            yield chunk


async def _run(mode: str, query: str, search_graph: _FakeGraph, analysis_graph: _FakeGraph) -> tuple[dict, AsyncMock]:
    """stream_agent를 끝까지 돌려 complete 이벤트의 result와 DB 저장 mock을 반환한다."""
    save = AsyncMock()
    with (
        patch.object(agent_service, "agent_graph", search_graph),
        patch.object(agent_service, "analyze_graph", analysis_graph),
        patch.object(agent_service, "_save_to_db", save),
    ):
        events = [json.loads(line.removeprefix("data: ")) async for line in stream_agent(mode, query, pdf_text="본문")]
    complete = next(e for e in events if e["event"] == "complete")
    return complete["result"], save


class ExtractPdfTextTest(unittest.TestCase):
    def test_pdf_without_text_layer_is_rejected(self):
        doc = fitz.open()
        doc.new_page()  # 텍스트 없는 빈 페이지 (스캔본과 같은 상황)
        with self.assertRaises(ValueError):
            extract_pdf_text(doc.tobytes())

    def test_pdf_with_text_is_extracted(self):
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "attention")
        self.assertEqual(extract_pdf_text(doc.tobytes()), "attention")


class GraphShapeTest(unittest.TestCase):
    def test_search_graph_has_no_analysis_nodes(self):
        nodes = set(agent_graph.get_graph().nodes)
        self.assertLessEqual({"planner", "researcher", "trend_analyzer"}, nodes)
        self.assertFalse({"analyzer", "coder", "reviewer"} & nodes)

    def test_analyze_graph_has_no_search_nodes(self):
        nodes = set(analyze_graph.get_graph().nodes)
        self.assertLessEqual({"analyzer", "coder", "reviewer"}, nodes)
        self.assertFalse({"planner", "researcher"} & nodes)


class StreamAgentModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_mode_skips_search_and_reports_filename(self):
        search = _FakeGraph([{"researcher": {"papers": [{"title": "다른 논문"}]}}])
        analysis = _FakeGraph([{"analyzer": {"paper_summary": "요약"}}])

        result, save = await _run("pdf", "my_paper.pdf", search, analysis)

        self.assertFalse(search.called)
        self.assertTrue(analysis.called)
        self.assertEqual(result["papers"], [])
        self.assertEqual(result["uploaded_filename"], "my_paper.pdf")
        self.assertEqual(result["paper_summary"], "요약")
        # 분석 이력은 계속 저장한다 (search_only=False)
        self.assertFalse(save.await_args.kwargs["search_only"])

    async def test_search_and_trend_modes_still_use_search_graph(self):
        for mode in ("search", "trend"):
            with self.subTest(mode=mode):
                search = _FakeGraph([{"researcher": {"papers": [{"title": "p"}]}}])
                analysis = _FakeGraph([])

                result, save = await _run(mode, "transformer", search, analysis)

                self.assertTrue(search.called)
                self.assertFalse(analysis.called)
                self.assertEqual(result["papers"], [{"title": "p"}])
                self.assertNotIn("uploaded_filename", result)
                self.assertTrue(save.await_args.kwargs["search_only"])


class SaveToDbTest(unittest.IsolatedAsyncioTestCase):
    async def _save(self, mode: str, search_only: bool) -> tuple[AsyncMock, AsyncMock]:
        history, analysis = AsyncMock(), AsyncMock()
        with (
            patch.object(agent_service, "AsyncSessionLocal"),
            patch.object(agent_service.crud_search_history, "create_search_history", history),
            patch.object(agent_service.crud_analysis, "create_analysis_result", analysis),
        ):
            agent_service.AsyncSessionLocal.return_value.__aenter__.return_value = AsyncMock()
            await agent_service._save_to_db(mode, "q", {"papers": []}, search_only=search_only)
        return history, analysis

    async def test_pdf_mode_saves_analysis_but_no_search_history(self):
        history, analysis = await self._save("pdf", search_only=False)
        history.assert_not_awaited()
        analysis.assert_awaited_once()

    async def test_search_mode_still_saves_search_history(self):
        history, analysis = await self._save("search", search_only=True)
        history.assert_awaited_once()
        analysis.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
