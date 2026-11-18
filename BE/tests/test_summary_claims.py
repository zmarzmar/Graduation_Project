"""요약 문장별 원문 인용 — 인용이 원문 페이지에 실제로 있는지만 확인한다 (LLM 없이)."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.nodes import analyzer
from agents.nodes.analyzer import _check_summary_claims

PAGES = [
    "Title page.",
    "",  # 빈 페이지 — 번호는 유지된다
    "We propose LoRA, which freezes the pretrained weights and injects trainable rank decomposition matrices.",
    "This suggests the update matrix could have a very small intrinsic rank.",
]


def _claim(text: str, *cites: tuple[int, str]) -> dict:
    return {"text": text, "citations": [{"page": page, "quote": quote} for page, quote in cites]}


class CheckSummaryClaimsTest(unittest.TestCase):
    def test_exact_quote_on_cited_page_is_verified(self) -> None:
        [claim] = _check_summary_claims([_claim("LoRA를 제안한다.", (3, "We propose LoRA, which freezes the pretrained weights"))], PAGES)
        self.assertTrue(claim["verified"])
        self.assertEqual(claim["citations"], [{"page": 3, "quote": "We propose LoRA, which freezes the pretrained weights"}])

    def test_wrong_page_number_uses_the_page_where_quote_is(self) -> None:
        # 모델이 인용문은 그대로 옮기고 페이지만 틀린 경우 (2026-11-18 AVQ 측정에서 관측)
        [claim] = _check_summary_claims([_claim("저랭크다.", (3, "the update matrix could have a very small intrinsic rank"))], PAGES)
        self.assertTrue(claim["verified"])
        self.assertEqual(claim["citations"][0]["page"], 4)

    def test_altered_quote_is_kept_but_unverified(self) -> None:
        # 단어 하나를 넣은 인용 ("suggests that the") — 원문과 다른 문장이다. 문장은 남기고 표시만 한다
        [claim] = _check_summary_claims([_claim("저랭크다.", (4, "This suggests that the update matrix could have"))], PAGES)
        self.assertFalse(claim["verified"])
        self.assertEqual(claim["citations"], [])
        self.assertEqual(claim["text"], "저랭크다.")

    def test_partly_valid_claim_is_unverified_but_shows_valid_quote(self) -> None:
        [claim] = _check_summary_claims(
            [_claim("섞임.", (3, "We propose LoRA, which freezes"), (3, "a sentence that does not exist anywhere"))], PAGES)
        self.assertFalse(claim["verified"])
        self.assertEqual(len(claim["citations"]), 1)

    def test_claim_without_citations_is_unverified(self) -> None:
        [claim] = _check_summary_claims([_claim("출처 없음.")], PAGES)
        self.assertFalse(claim["verified"])

    def test_malformed_items_are_skipped_or_dropped(self) -> None:
        raw = ["문자열", {"text": ""}, {"text": "정상", "citations": [{"page": "3", "quote": "We propose LoRA, which"}, "x"]}]
        claims = _check_summary_claims(raw, PAGES)
        self.assertEqual([c["text"] for c in claims], ["정상"])
        self.assertFalse(claims[0]["verified"])
        self.assertEqual(_check_summary_claims(None, PAGES), [])
        self.assertEqual(_check_summary_claims({"text": "x"}, PAGES), [])

    def test_short_quote_is_not_evidence(self) -> None:
        # Q&A와 같은 최소 길이 — 너무 짧은 인용은 어디에나 있다
        [claim] = _check_summary_claims([_claim("짧음.", (3, "LoRA"))], PAGES)
        self.assertFalse(claim["verified"])


def _fake(payload: dict) -> object:
    class _LLM:
        def __init__(self) -> None:
            self.messages: list = []

        async def ainvoke(self, messages: list) -> SimpleNamespace:
            self.messages = messages
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    return _LLM()


class AnalyzerNodeClaimsTest(unittest.IsolatedAsyncioTestCase):
    PAYLOAD = {
        "summary_claims": [
            _claim("LoRA를 제안한다.", (3, "We propose LoRA, which freezes the pretrained weights")),
            _claim("저랭크다.", (4, "This suggests that the update matrix could have")),
        ],
        "review": {}, "key_formulas": [],
    }

    async def test_pdf_mode_sends_numbered_pages_and_returns_claims(self) -> None:
        fake = _fake(self.PAYLOAD)
        with patch.object(analyzer, "_llm", fake):
            result = await analyzer.analyzer_node({"pdf_text": "\n".join(PAGES), "pdf_pages": PAGES, "papers": []})
        self.assertIsNone(result["error"])
        self.assertEqual(result["paper_summary"], "LoRA를 제안한다. 저랭크다.")
        self.assertEqual([c["verified"] for c in result["summary_claims"]], [True, False])
        sent = fake.messages[1].content
        self.assertIn('<page number="3">', sent)
        self.assertNotIn('<page number="2">', sent)  # 빈 페이지는 보내지 않지만 뒤 번호는 원본 그대로

    async def test_abstract_mode_has_no_claims(self) -> None:
        # 원문이 없으면 인용을 확인할 수 없다 — 요약 문자열만 쓰고 claims는 비운다
        with patch.object(analyzer, "_llm", _fake(self.PAYLOAD)):
            result = await analyzer.analyzer_node({"papers": [{"title": "T", "abstract": "A"}]})
        self.assertEqual(result["paper_summary"], "LoRA를 제안한다. 저랭크다.")
        self.assertEqual(result["summary_claims"], [])

    async def test_pdf_summary_without_claims_is_marked_unverified(self) -> None:
        # Codex 리뷰: 원문이 있는데 인용 없이 요약만 오면 기존 기록처럼 표시 없이 보이던 문제
        for payload in ({"summary": "요약", "review": {}, "key_formulas": []},
                        {"summary": "요약", "summary_claims": [], "review": {}, "key_formulas": []}):
            with self.subTest(payload=payload), patch.object(analyzer, "_llm", _fake(payload)):
                result = await analyzer.analyzer_node({"pdf_text": "x", "pdf_pages": ["x"], "papers": []})
            self.assertEqual(result["paper_summary"], "요약")
            self.assertEqual(result["summary_claims"], [{"text": "요약", "citations": [], "verified": False}])

    async def test_non_list_citations_do_not_stop_the_analysis(self) -> None:
        # Codex 리뷰: citations가 숫자·불리언이면 TypeError로 그래프가 멈췄다
        for bad in (1, True, "p.3", {"page": 3}):
            payload = {"summary_claims": [{"text": "요약", "citations": bad}], "review": {}, "key_formulas": []}
            with self.subTest(citations=bad), patch.object(analyzer, "_llm", _fake(payload)):
                result = await analyzer.analyzer_node({"pdf_text": "\n".join(PAGES), "pdf_pages": PAGES, "papers": []})
            self.assertIsNone(result["error"])
            self.assertEqual(result["summary_claims"], [{"text": "요약", "citations": [], "verified": False}])


if __name__ == "__main__":
    unittest.main()
