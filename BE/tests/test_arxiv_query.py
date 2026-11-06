"""arXiv 검색은 초록 구절 검색이다 — 따옴표 없는 `all:a b`는 arXiv가 "a OR b"로 해석한다."""

import unittest
from unittest.mock import patch

import httpx

from services import arxiv_service


class ArxivQueryTest(unittest.IsolatedAsyncioTestCase):
    def test_phrase_in_abstract(self) -> None:
        self.assertEqual(arxiv_service._build_search_query("video generation"), 'abs:"video generation"')
        self.assertEqual(arxiv_service._build_search_query('  "LoRA"  fine-tuning '), 'abs:"LoRA fine-tuning"')

    async def test_request_uses_phrase_query(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")

        real_client = httpx.AsyncClient
        arxiv_service._cache.clear()
        with patch.object(
            arxiv_service.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handler)),
        ):
            await arxiv_service.search_papers("video generation", max_results=3, recent_months=6)

        self.assertEqual(seen[0].url.params["search_query"], 'abs:"video generation"')


if __name__ == "__main__":
    unittest.main()
