"""Researcher 노드 — 선택 기능인 S2 인용 수 보완이 실패해도 수집한 논문을 버리지 않는다."""

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from agents.nodes import researcher
from schemas.paper import PaperResult


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


if __name__ == "__main__":
    unittest.main()
