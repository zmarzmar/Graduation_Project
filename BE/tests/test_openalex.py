"""OpenAlex 키는 헤더로만 보내고 요청 URL에는 넣지 않는다 (URL은 로그에 남는다)."""

import unittest
from unittest.mock import patch

import httpx

from services import openalex_service


class OpenAlexKeyTest(unittest.IsolatedAsyncioTestCase):
    async def test_key_goes_in_header_not_url(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"results": []})

        real_client = httpx.AsyncClient
        with (
            patch.object(openalex_service.settings, "openalex_api_key", "test-key-123"),
            patch.object(
                openalex_service.httpx,
                "AsyncClient",
                lambda **kw: real_client(transport=httpx.MockTransport(handler)),
            ),
        ):
            await openalex_service.search_papers("mixture of experts")

        self.assertEqual(seen[0].headers["authorization"], "Bearer test-key-123")
        self.assertNotIn("test-key-123", str(seen[0].url))

    async def test_failure_log_does_not_contain_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.LocalProtocolError(f"Illegal header value {request.headers['authorization']!r}")

        real_client = httpx.AsyncClient
        with (
            patch.object(openalex_service.settings, "openalex_api_key", "test-key-123\n"),
            patch.object(
                openalex_service.httpx,
                "AsyncClient",
                lambda **kw: real_client(transport=httpx.MockTransport(handler)),
            ),
            self.assertLogs(openalex_service.logger, level="WARNING") as logs,
        ):
            result = await openalex_service.search_papers("q")

        self.assertEqual(result, [])
        self.assertFalse([line for line in logs.output if "test-key-123" in line], logs.output)

    def test_key_whitespace_is_stripped(self) -> None:
        with patch.object(openalex_service.settings, "openalex_api_key", " test-key-123\n"):
            self.assertEqual(openalex_service._build_headers()["Authorization"], "Bearer test-key-123")

    def test_no_key_no_auth_header(self) -> None:
        with patch.object(openalex_service.settings, "openalex_api_key", ""):
            self.assertNotIn("Authorization", openalex_service._build_headers())


if __name__ == "__main__":
    unittest.main()
