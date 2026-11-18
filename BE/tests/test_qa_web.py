"""논문 Q&A — 약어 보충 검색어와 웹 검색 보충 답변 (OpenAI·DB 없이)."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents import qa, web_answer
from agents.qa import Citation, Claim, QaDraft
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
        result, web = await self._ask(rag_service.no_evidence(), history=[{"question": "멀티브랜치가 뭔데?", "answer": "논문 답"}])
        self.assertEqual(result["web_answer"], WEB)
        self.assertFalse(result["answerable"])  # 논문 답변은 그대로 '근거 없음'
        # 제목(파일명)도, 논문에서 얻은 직전 답도 보내지 않는다 (업로드 PDF의 내용일 수 있다 — Codex 리뷰)
        web.assert_awaited_once_with("CLS 토큰이 뭔데?", "멀티브랜치가 뭔데?", "")

    async def test_previous_web_answer_is_passed_without_its_marker(self):
        history = [{"question": "정규화 방법?", "answer": rag_service.WEB_ANSWER_PREFIX + "LayerNorm과 BatchNorm이 있다"}]
        _, web = await self._ask(rag_service.no_evidence(), history=history)
        web.assert_awaited_once_with("CLS 토큰이 뭔데?", "정규화 방법?", "LayerNorm과 BatchNorm이 있다")

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

    async def test_request_tells_the_model_to_refuse_non_academic_questions(self):
        # Codex 리뷰: 아래 테스트는 응답만 고정한다 — 실제 요청에 거절 규칙과 웹 검색 도구가 들어가는지도 본다
        create = AsyncMock(return_value=_response("OFF_TOPIC", []))
        with patch.object(web_answer._client.responses, "create", create):
            await web_answer.answer_from_web("주가 전망 알려줘")
        kwargs = create.await_args.kwargs
        self.assertIn("OFF_TOPIC", kwargs["instructions"])
        self.assertIn("주가", kwargs["instructions"])
        self.assertEqual(kwargs["tools"], [{"type": "web_search"}])

    async def test_previous_answer_is_sent_as_reference(self):
        # Codex 리뷰: 직전 웹 답변에만 나온 용어("두 번째 방법")를 후속 질문이 가리킬 수 있다
        create = AsyncMock(return_value=_response("OFF_TOPIC", []))
        with patch.object(web_answer._client.responses, "create", create):
            await web_answer.answer_from_web("두 번째 방법의 단점은?", "정규화 방법 알려줘", "LayerNorm과 BatchNorm이 있다")
        self.assertIn("LayerNorm과 BatchNorm이 있다", create.await_args.kwargs["input"])

    async def test_off_topic_question_gives_none(self):
        # 주가·날씨 같은 질문에 모델이 OFF_TOPIC이라고만 답하면 (출처가 붙어 있어도) 웹 답변을 붙이지 않는다
        self.assertIsNone(await self._call(_response("OFF_TOPIC", [_cite("https://news.example.com")])))


class RetryWhenEveryClaimFailsTest(unittest.IsolatedAsyncioTestCase):
    """모델 답의 문장이 모두 인용 검증에서 떨어지면 한 번만 다시 묻는다 (2026-11-18 운영: √dₖ 질문 4회 중 1회)."""

    PASSAGES = [Passage(chunk_index=0, page=4, text="We suspect that for large values of dk, the dot products grow large.", distance=0.1)]
    GOOD = QaDraft(answerable=True, claims=[Claim(text="dk가 크면 내적이 커진다.", citations=[
        Citation(passage=1, quote="for large values of dk, the dot products grow large")])])
    BAD = QaDraft(answerable=True, claims=[Claim(text="내적이 커진다.", citations=[
        Citation(passage=1, quote="for large values of d_k the dot products grow")])])  # 인용문을 바꿔 옮김

    async def _run(self, *drafts: QaDraft) -> tuple[dict, int]:
        calls = {"n": 0}

        class _LLM:
            async def ainvoke(self, messages):
                calls["n"] += 1
                return drafts[min(calls["n"], len(drafts)) - 1]

        with patch.object(qa, "_llm", _LLM()):
            result = await qa.answer_question("왜 √dₖ로 나누나요?", self.PASSAGES)
        return result, calls["n"]

    async def test_all_dropped_then_retried_once(self):
        result, calls = await self._run(self.BAD, self.GOOD)
        self.assertEqual(calls, 2)
        self.assertTrue(result["answerable"])

    async def test_gives_up_after_the_second_try(self):
        result, calls = await self._run(self.BAD, self.BAD, self.GOOD)
        self.assertEqual(calls, 2)
        self.assertFalse(result["answerable"])
        self.assertEqual(result["dropped_claims"], 1)

    async def test_retry_failure_keeps_the_first_result(self):
        # Codex 리뷰: 재시도 호출만 실패하면 502가 아니라 첫 결과('근거 없음' + 떨어진 수)를 돌려준다
        calls = {"n": 0}
        bad = self.BAD

        class _LLM:
            async def ainvoke(self, messages):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise TimeoutError()
                return bad

        with patch.object(qa, "_llm", _LLM()):
            result = await qa.answer_question("왜 √dₖ로 나누나요?", self.PASSAGES)
        self.assertEqual(calls["n"], 2)
        self.assertFalse(result["answerable"])
        self.assertEqual(result["dropped_claims"], 1)

    async def test_no_retry_when_answered_or_refused(self):
        self.assertEqual((await self._run(self.GOOD))[1], 1)
        self.assertEqual((await self._run(QaDraft(answerable=False)))[1], 1)


class SourceTitleTest(unittest.TestCase):
    def test_arxiv_links_get_a_clean_title_and_long_titles_are_cut(self):
        self.assertEqual(web_answer._title("Provided proper attribution is provided, Google hereby grants",
                                           "https://arxiv.org/pdf/1706.03762"), "arXiv 1706.03762")
        self.assertEqual(web_answer._title("BERT", "https://arxiv.org/abs/1810.04805v2"), "arXiv 1810.04805")
        self.assertEqual(web_answer._title("license", "https://arxiv.org/pdf/quant-ph/9508027"), "arXiv quant-ph/9508027")
        self.assertEqual(len(web_answer._title("x" * 300, "https://example.com")), 100)
        self.assertEqual(web_answer._title("  LayerNorm \n docs ", "https://pytorch.org"), "LayerNorm docs")


if __name__ == "__main__":
    unittest.main()
