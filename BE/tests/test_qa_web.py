"""논문 Q&A — 약어 보충 검색어와 웹 검색 보충 답변 (OpenAI·DB 없이)."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents import web_answer
from services import rag_service
from services.rag_service import Passage, _key_terms


class KeyTermsTest(unittest.TestCase):
    def test_acronyms_and_versioned_names_only(self):
        self.assertEqual(_key_terms("CLS 토큰이 뭔데?"), ["CLS"])
        self.assertEqual(_key_terms("What is the CLS token in BERT?"), ["CLS", "BERT"])  # 'What'은 일반 단어
        self.assertEqual(_key_terms("LoRA와 RoPE, GPT-3 차이"), ["LoRA", "RoPE", "GPT-3"])
        self.assertEqual(_key_terms("attention이 뭐야? Transformer는?"), [])
        self.assertEqual(_key_terms("x y Q K"), [])  # 한 글자는 어디에나 있다

    def test_at_most_three_terms_without_duplicates(self):
        self.assertEqual(_key_terms("CLS CLS GQA RMSNorm SwiGLU"), ["CLS", "GQA", "RMSNorm"])


DOC = SimpleNamespace(id=1, title="SSF-SET")
PASSAGE = Passage(chunk_index=0, page=3, text="t", distance=0.2)
WEB = {"text": "CLS 토큰은 시퀀스 앞에 붙이는 학습 가능한 토큰이다.", "sources": [{"title": "BERT", "url": "https://example.com"}]}


class WebFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def _ask(self, answer: dict, enabled: bool = True, history: list[dict] | None = None) -> tuple[dict, AsyncMock]:
        web = AsyncMock(return_value=WEB)
        with patch.object(rag_service, "ensure_indexed", AsyncMock(return_value="job")), \
                patch.object(rag_service, "retrieve", AsyncMock(return_value=[PASSAGE])), \
                patch.object(rag_service, "answer_question", AsyncMock(return_value=answer)), \
                patch.object(rag_service, "answer_from_web", web), \
                patch.object(rag_service.settings, "qa_web_fallback", enabled):
            return await rag_service.ask(DOC, "CLS 토큰이 뭔데?", history), web

    async def test_no_evidence_gets_a_web_answer(self):
        result, web = await self._ask(rag_service.no_evidence(), history=[{"question": "멀티브랜치가 뭔데?", "answer": "…"}])
        self.assertEqual(result["web_answer"], WEB)
        self.assertFalse(result["answerable"])  # 논문 답변은 그대로 '근거 없음'
        web.assert_awaited_once_with("CLS 토큰이 뭔데?", "멀티브랜치가 뭔데?")  # 제목(파일명)은 보내지 않는다

    async def test_answered_from_paper_does_not_search_the_web(self):
        result, web = await self._ask({"answerable": True, "answer": "a", "claims": [], "citations": [],
                                       "dropped_citations": 0, "dropped_claims": 0})
        self.assertIsNone(result["web_answer"])
        web.assert_not_awaited()

    async def test_switched_off_by_config(self):
        result, web = await self._ask(rag_service.no_evidence(), enabled=False)
        self.assertIsNone(result["web_answer"])
        web.assert_not_awaited()


def _response(text: str, annotations: list, status: str = "completed") -> SimpleNamespace:
    part = SimpleNamespace(annotations=annotations)
    return SimpleNamespace(status=status, output_text=text,
                           output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message", content=[part])])


def _cite(url: str, title: str = "t") -> SimpleNamespace:
    return SimpleNamespace(type="url_citation", url=url, title=title)


class AnswerFromWebTest(unittest.IsolatedAsyncioTestCase):
    async def _call(self, response=None, error: Exception | None = None) -> dict | None:
        create = AsyncMock(return_value=response, side_effect=error)
        with patch.object(web_answer._client.responses, "create", create):
            return await web_answer.answer_from_web("CLS?")

    async def test_strips_inline_links_and_keeps_http_sources(self):
        text = "CLS는 **대표 토큰**이다. ([huggingface.co](https://huggingface.co/x?utm_source=openai))  \n분류에 쓴다."
        result = await self._call(_response(text, [
            _cite("https://huggingface.co/x", "HF"), _cite("https://huggingface.co/x", "HF"),  # 중복
            _cite("javascript:alert(1)"), SimpleNamespace(type="file_citation", url="https://a"),
        ]))
        self.assertEqual(result["text"], "CLS는 대표 토큰이다.  \n분류에 쓴다.")
        self.assertEqual(result["sources"], [{"title": "HF", "url": "https://huggingface.co/x"}])

    async def test_failure_truncation_or_no_source_gives_none(self):
        self.assertIsNone(await self._call(error=TimeoutError()))
        self.assertIsNone(await self._call(_response("답", [_cite("https://a")], status="incomplete")))
        self.assertIsNone(await self._call(_response("출처 없는 답", [])))


if __name__ == "__main__":
    unittest.main()
