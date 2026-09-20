"""논문 전문 공유·길이 한도·페이지 보존 테스트.

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import fitz
import httpx
from fastapi.testclient import TestClient

from agents.nodes import coder, reviewer
from agents.paper_context import paper_source_context
from services import agent_service
from services.agent_service import PaperTooLongError, extract_pdf_pages, join_pages, stream_agent, stream_analyze

# 예전 Reviewer가 잘라내던 8,000자보다 훨씬 뒤에 놓이는 표식
_LATE_MARKER = "METHODS-SECTION-MARKER"


def _pdf(page_texts: list[str]) -> bytes:
    """페이지별 텍스트로 PDF를 만든다. 빈 문자열은 빈 페이지가 된다."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    return doc.tobytes()


class _FakeLLM:
    """ainvoke로 받은 사용자 메시지를 기록한다."""

    def __init__(self, content: str):
        self.content = content
        self.user_message = ""

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        self.user_message = messages[-1].content
        return SimpleNamespace(content=self.content)


class _CapturingGraph:
    """그래프에 전달된 초기 상태를 기록한다."""

    def __init__(self):
        self.state: dict | None = None

    async def astream(self, state: dict, stream_mode: str = "updates"):
        self.state = state
        yield {"analyzer": {"paper_summary": "요약"}}


def _serve_pdf(pdf: bytes):
    """다운로드 경로가 네트워크 대신 주어진 PDF를 받도록 httpx 클라이언트를 바꾼다."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=pdf))
    real_client = httpx.AsyncClient
    return patch.object(agent_service.httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw))


async def _events(stream) -> list[dict]:
    return [json.loads(line.removeprefix("data: ")) async for line in stream]


class SharedFullTextTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = {
            "pdf_text": "intro " * 5000 + _LATE_MARKER + " appendix " * 100,
            "papers": [],
            "generated_code": "import torch",
            "iteration_count": 1,
        }
        self.assertGreater(self.state["pdf_text"].index(_LATE_MARKER), 8000)

    async def test_reviewer_receives_text_beyond_8000_chars(self):
        llm = _FakeLLM('{"passed": true, "feedback": "ok", "issues": [], "suggestions": []}')
        with patch.object(reviewer, "_llm", llm):
            await reviewer.reviewer_node(self.state)
        self.assertIn(_LATE_MARKER, llm.user_message)

    async def test_coder_and_reviewer_share_the_same_paper_source(self):
        source = paper_source_context(self.state)
        coder_llm = _FakeLLM("```python\nimport torch\n```")
        reviewer_llm = _FakeLLM('{"passed": true, "feedback": "ok", "issues": [], "suggestions": []}')
        with patch.object(coder, "_llm", coder_llm), patch.object(reviewer, "_llm", reviewer_llm):
            await coder.coder_node({**self.state, "iteration_count": 0})
            await reviewer.reviewer_node(self.state)
        # 논문 원문 부분이 자르지 않은 채 양쪽에 그대로 들어간다 (요약·이전 코드 등 나머지 입력은 노드별로 다름)
        self.assertIn(self.state["pdf_text"], source)
        self.assertIn(source, coder_llm.user_message)
        self.assertIn(source, reviewer_llm.user_message)


class CountTokensTest(unittest.TestCase):
    # 바이트 / 3 근사가 실제의 절반 수준으로 적게 잡던 종류의 텍스트
    SYMBOLS = "∑ᵢ αᵢ·xᵢ ≤ ‖W‖₂ ∀θ∈Θ ⊗ 0.137 42.5 | " * 50

    def test_symbol_heavy_text_is_counted_with_the_real_tokenizer(self):
        try:
            agent_service._token_encoding()
        except Exception as e:  # 인코딩 파일을 받을 수 없는 환경
            self.skipTest(f"tokenizer unavailable: {e}")
        self.assertGreater(agent_service.count_tokens(self.SYMBOLS), len(self.SYMBOLS.encode()) // 3)

    def test_falls_back_to_byte_estimate_when_tokenizer_is_unavailable(self):
        with patch.object(agent_service, "_token_encoding", side_effect=OSError("no network")):
            self.assertEqual(agent_service.count_tokens(self.SYMBOLS), len(self.SYMBOLS.encode()) // 3)


class PaperTooLongTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 한도를 낮춰 작은 PDF로 초과 상황을 만든다
        limit = patch.object(agent_service.settings, "max_paper_tokens", 5)
        limit.start()
        self.addCleanup(limit.stop)
        self.pdf = _pdf(["a long paper body that exceeds the tiny limit"])

    def test_extraction_raises_instead_of_truncating(self):
        with self.assertRaises(PaperTooLongError) as ctx:
            extract_pdf_pages(self.pdf)
        self.assertIn("한도", str(ctx.exception))

    def test_upload_returns_413_with_clear_message(self):
        from main import app

        with patch.object(agent_service, "stream_agent") as stream:
            response = TestClient(app).post(
                "/api/v1/agent/pdf", files={"file": ("long.pdf", self.pdf, "application/pdf")}
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("논문이 너무 깁니다", response.json()["detail"])
        stream.assert_not_called()  # 분석을 시작하지 않는다

    async def test_download_path_reports_too_long_not_download_failure(self):
        graph = _CapturingGraph()
        with _serve_pdf(self.pdf), patch.object(agent_service, "analyze_graph", graph):
            events = await _events(stream_analyze({"arxiv_id": "1706.03762", "abstract": "abs"}, "q"))

        fallback = next(e for e in events if e["event"] == "pdf_fallback_required")
        self.assertEqual(fallback["reason"], "paper_too_long")
        self.assertIn("논문이 너무 깁니다", fallback["message"])
        self.assertNotIn("다운로드하지 못했습니다", fallback["message"])
        self.assertIsNone(graph.state)  # 사용자 동의 전에는 분석을 시작하지 않는다


class PagePreservationTest(unittest.IsolatedAsyncioTestCase):
    PAGES = ["first page", "", "third page"]  # 가운데는 빈 페이지

    def assert_pages(self, pages: list[str]):
        self.assertEqual([p.strip() for p in pages], self.PAGES)

    def test_blank_pages_keep_their_position(self):
        self.assert_pages(extract_pdf_pages(_pdf(self.PAGES)))

    def test_fully_blank_pdf_is_still_rejected(self):
        with self.assertRaises(ValueError):
            extract_pdf_pages(_pdf(["", ""]))

    def test_joined_text_matches_previous_full_text_format(self):
        pdf = _pdf(self.PAGES)
        previous = "\n".join(page.get_text() for page in fitz.open(stream=pdf, filetype="pdf")).strip()
        self.assertEqual(join_pages(extract_pdf_pages(pdf)), previous)

    async def test_upload_path_passes_pages_to_the_graph(self):
        graph = _CapturingGraph()
        pages = extract_pdf_pages(_pdf(self.PAGES))
        with patch.object(agent_service, "analyze_graph", graph), patch.object(agent_service, "_save_to_db", AsyncMock()):
            await _events(stream_agent("pdf", "upload.pdf", pdf_pages=pages))
        self.assert_pages(graph.state["pdf_pages"])
        self.assertEqual(graph.state["pdf_text"], join_pages(pages))

    async def test_download_path_passes_pages_to_the_graph(self):
        graph = _CapturingGraph()
        with (
            _serve_pdf(_pdf(self.PAGES)),
            patch.object(agent_service, "analyze_graph", graph),
            patch.object(agent_service, "_save_analyze_to_db", AsyncMock()),
        ):
            await _events(stream_analyze({"arxiv_id": "1706.03762", "abstract": "abs"}, "q"))
        self.assert_pages(graph.state["pdf_pages"])
        self.assertEqual(graph.state["pdf_text"], join_pages(graph.state["pdf_pages"]))


if __name__ == "__main__":
    unittest.main()
