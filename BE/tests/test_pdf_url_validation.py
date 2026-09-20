"""PDF 다운로드 URL 검증(SSRF 방지) 테스트.

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import unittest
from unittest.mock import patch

import fitz
import httpx

from services import agent_service
from services.agent_service import (
    _download_first_available_pdf_pages,
    _validate_pdf_url,
    download_pdf_pages,
)

_BLOCKED_URLS = [
    "http://arxiv.org/pdf/1706.03762",           # https 아님
    "ftp://arxiv.org/pdf/1706.03762",
    "file:///etc/passwd",
    "https://evilarxiv.org/pdf/1706.03762",      # 접미사 일치 우회
    "https://arxiv.org.evil.com/pdf/1706.03762", # 접두사 일치 우회
    "https://arxiv.org:8080/pdf/1706.03762",     # 포트 우회
    "https://user@arxiv.org/pdf/1706.03762",     # userinfo
    "https://arxiv.org@evil.com/pdf/1706.03762", # userinfo로 호스트 위장
    "https://arxiv.org\\@evil.com/pdf/x",        # 파서 해석 차이 노림
    "https://localhost/pdf/x",
    "https://127.0.0.1/pdf/x",
    "https://169.254.169.254/latest/meta-data/", # 클라우드 메타데이터
    "https://[::1]/pdf/x",
    "https://www.semanticscholar.org/reader/x",  # 의도적 축소: arXiv 외 출처
    "/pdf/1706.03762",                           # 스킴·호스트 없음
    "",
]

_ALLOWED_URLS = [
    "https://arxiv.org/pdf/1706.03762",
    "https://ARXIV.org/pdf/1706.03762v7.pdf",
    "https://www.arxiv.org/pdf/1706.03762",
    "https://export.arxiv.org/pdf/1706.03762",
    "https://arxiv.org:443/pdf/1706.03762",
]


def _pdf_bytes(text: str = "attention is all you need") -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


class _Recorder:
    """MockTransport 핸들러 — 실제로 나간 요청 URL을 전부 기록한다."""

    def __init__(self, routes: dict[str, httpx.Response]):
        self.routes = routes
        self.requested: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        return self.routes.get(url, httpx.Response(404))

    def patch_client(self):
        real_client = httpx.AsyncClient
        return patch.object(
            agent_service.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(self), **kw),
        )


def _redirect(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


class ValidatePdfUrlTest(unittest.TestCase):
    def test_blocked(self):
        for url in _BLOCKED_URLS:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_pdf_url(url)

    def test_allowed(self):
        for url in _ALLOWED_URLS:
            with self.subTest(url=url):
                _validate_pdf_url(url)


class DownloadPdfTextTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_url_sends_no_request(self):
        rec = _Recorder({})
        with rec.patch_client():
            for url in _BLOCKED_URLS:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    await download_pdf_pages(url)
        self.assertEqual(rec.requested, [])

    async def test_downloads_allowed_pdf(self):
        rec = _Recorder({"https://arxiv.org/pdf/1": httpx.Response(200, content=_pdf_bytes())})
        with rec.patch_client():
            pages = await download_pdf_pages("https://arxiv.org/pdf/1")
        self.assertIn("attention", pages[0])

    async def test_redirect_to_disallowed_host_is_never_requested(self):
        for target in ("https://evil.com/x.pdf", "http://169.254.169.254/", "http://localhost:8000/health"):
            rec = _Recorder({"https://arxiv.org/pdf/1": _redirect(target)})
            with self.subTest(target=target), rec.patch_client():
                with self.assertRaises(ValueError):
                    await download_pdf_pages("https://arxiv.org/pdf/1")
                self.assertEqual(rec.requested, ["https://arxiv.org/pdf/1"])

    async def test_relative_redirect_is_resolved_and_followed(self):
        rec = _Recorder({
            "https://arxiv.org/pdf/1": _redirect("/pdf/1v2"),
            "https://arxiv.org/pdf/1v2": httpx.Response(200, content=_pdf_bytes()),
        })
        with rec.patch_client():
            pages = await download_pdf_pages("https://arxiv.org/pdf/1")
        self.assertIn("attention", pages[0])
        self.assertEqual(rec.requested, ["https://arxiv.org/pdf/1", "https://arxiv.org/pdf/1v2"])

    async def test_scheme_relative_redirect_to_other_host_is_blocked(self):
        rec = _Recorder({"https://arxiv.org/pdf/1": _redirect("//evil.com/x.pdf")})
        with rec.patch_client(), self.assertRaises(ValueError):
            await download_pdf_pages("https://arxiv.org/pdf/1")
        self.assertEqual(rec.requested, ["https://arxiv.org/pdf/1"])

    async def test_three_redirects_are_followed(self):
        rec = _Recorder({
            "https://arxiv.org/pdf/0": _redirect("/pdf/1"),
            "https://arxiv.org/pdf/1": _redirect("/pdf/2"),
            "https://arxiv.org/pdf/2": _redirect("https://export.arxiv.org/pdf/3"),
            "https://export.arxiv.org/pdf/3": httpx.Response(200, content=_pdf_bytes()),
        })
        with rec.patch_client():
            await download_pdf_pages("https://arxiv.org/pdf/0")
        self.assertEqual(len(rec.requested), 4)

    async def test_fourth_redirect_is_not_followed(self):
        rec = _Recorder({f"https://arxiv.org/pdf/{i}": _redirect(f"/pdf/{i + 1}") for i in range(10)})
        with rec.patch_client(), self.assertRaises(ValueError):
            await download_pdf_pages("https://arxiv.org/pdf/0")
        self.assertEqual(len(rec.requested), 4)

    async def test_paper_with_only_disallowed_urls_sends_no_request(self):
        rec = _Recorder({})
        paper = {"pdf_url": "https://evilarxiv.org/abs/1706.03762", "url": "http://localhost:8000/health"}
        with rec.patch_client(), self.assertRaises(ValueError):
            await _download_first_available_pdf_pages(paper)
        self.assertEqual(rec.requested, [])


if __name__ == "__main__":
    unittest.main()
