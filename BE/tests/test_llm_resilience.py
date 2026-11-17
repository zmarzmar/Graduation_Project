"""LLM 호출 시간 제한 (실제 OpenAI를 부르지 않는다).

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_openai import ChatOpenAI

from agents import qa
from agents.nodes import analyzer, coder, planner, reviewer, trend_analyzer
from core.config import settings
from services import rag_service


class ClientLimitsTest(unittest.TestCase):
    """모든 LLM 클라이언트가 요청당 시간 제한과 재시도 횟수를 갖는다 (없으면 SDK 기본 600초 × 3회)."""

    def test_chat_clients_have_timeout_and_retries(self) -> None:
        cases = {
            coder: settings.llm_timeout_reasoning_seconds,
            reviewer: settings.llm_timeout_reasoning_seconds,
            analyzer: settings.llm_timeout_analyzer_seconds,
            planner: settings.llm_timeout_short_seconds,
            trend_analyzer: settings.llm_timeout_short_seconds,
        }
        for module, timeout in cases.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(module._llm.request_timeout, timeout)
                self.assertEqual(module._llm.max_retries, settings.llm_max_retries)

    def test_qa_client_has_timeout_and_retries(self) -> None:
        # with_structured_output은 ChatOpenAI를 감싼 체인이다 — 안의 ChatOpenAI를 찾는다
        found = list({id(r): r for r in _walk(qa._llm) if isinstance(r, ChatOpenAI)}.values())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].request_timeout, settings.llm_timeout_short_seconds)
        self.assertEqual(found[0].max_retries, settings.llm_max_retries)

    def test_embedding_client_has_timeout_and_retries(self) -> None:
        rag_service._openai.cache_clear()
        client = rag_service._openai()
        self.assertEqual(client.timeout, settings.llm_timeout_short_seconds)
        self.assertEqual(client.max_retries, settings.llm_max_retries)


def _walk(runnable: object, depth: int = 0) -> list:
    if depth > 5:
        return []
    out = [runnable]
    for attr in ("first", "bound", "steps", "middle", "last"):
        child = getattr(runnable, attr, None)
        for c in child if isinstance(child, list) else [child]:
            if c is not None and c is not runnable:
                out += _walk(c, depth + 1)
    return out


class StalledServerTest(unittest.IsolatedAsyncioTestCase):
    """응답하지 않는 서버에 붙으면 시간 제한 뒤 노드의 기존 오류 경로로 끝난다 (2026-11-16: 요청 하나가 1,104초 매달림)."""

    async def asyncSetUp(self) -> None:
        async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await asyncio.sleep(30)  # 요청을 받기만 하고 답하지 않는다
            writer.close()

        self.server = await asyncio.start_server(stall, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self) -> None:
        self.server.close()

    async def test_coder_returns_error_after_timeout(self) -> None:
        llm = ChatOpenAI(model="gpt-5.4-mini", api_key="test", base_url=f"http://127.0.0.1:{self.port}/v1",
                         timeout=0.5, max_retries=0)
        started = time.perf_counter()
        with patch.object(coder, "_llm", llm):
            result = await coder.coder_node({"pdf_text": "paper", "papers": [], "iteration_count": 0})
        self.assertLess(time.perf_counter() - started, 10)
        self.assertIn("코드 생성 실패", result["error"])


if __name__ == "__main__":
    unittest.main()
