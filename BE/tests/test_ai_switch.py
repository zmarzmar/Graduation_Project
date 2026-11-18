"""AI 사용 중지 스위치 — 끄면 비용이 드는 요청(검색·분석·PDF·트렌드·Q&A)이 503, 관리자만 바꿀 수 있다.

설정 저장은 실제 Postgres로 본다 (ON CONFLICT). DB에 붙을 수 없으면 건너뛴다 — CI는 REQUIRE_DB_TESTS=1.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import core.dependencies as dependencies
from core.config import settings
from core.dependencies import get_current_user, get_db, require_ai_enabled
from models.analysis import AnalysisResult
from models.app_setting import AppSetting
from services import agent_service, ai_switch_service

MESSAGE = ai_switch_service.AI_DISABLED_MESSAGE


class AiSwitchDbTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(settings.database_url, poolclass=NullPool)
        try:
            self.conn = await self.engine.connect()
        except Exception as e:
            await self.engine.dispose()
            if os.environ.get("REQUIRE_DB_TESTS"):
                raise
            self.skipTest(f"database unavailable: {e}")
        self.outer = await self.conn.begin()
        # 서비스의 commit은 세이브포인트가 된다 → 마지막에 전부 롤백 (스위치 상태를 남기지 않는다)
        self.sessions = async_sessionmaker(bind=self.conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
        self.db: AsyncSession = self.sessions()
        await self.db.execute(delete(AppSetting).where(AppSetting.key == ai_switch_service.AI_ENABLED_KEY))

    async def asyncTearDown(self):
        await self.db.close()
        await self.outer.rollback()
        await self.conn.close()
        await self.engine.dispose()

    async def test_missing_row_means_enabled(self):
        self.assertTrue(await ai_switch_service.is_ai_enabled(self.db))
        status = await ai_switch_service.get_status(self.db)
        self.assertTrue(status["enabled"])
        self.assertIsNone(status["updated_at"])

    async def test_turn_off_then_on(self):
        # 행이 없을 때 만들고(INSERT), 있으면 바꾼다(ON CONFLICT UPDATE)
        off = await ai_switch_service.set_ai_enabled(self.db, False)
        self.assertFalse(off["enabled"])
        self.assertIsNotNone(off["updated_at"])
        self.assertFalse(await ai_switch_service.is_ai_enabled(self.db))
        on = await ai_switch_service.set_ai_enabled(self.db, True)
        self.assertTrue(on["enabled"])
        self.assertTrue(await ai_switch_service.is_ai_enabled(self.db))

    async def test_today_counts_include_deleted_records(self):
        before = (await ai_switch_service.get_status(self.db))["today_analyses"]
        self.db.add(AnalysisResult(mode="pdf", query="q", generated_code="", review_feedback="", review_passed=False,
                                   iteration_count=0, is_deleted=True))
        await self.db.flush()
        # 삭제한 기록도 이미 비용이 든 요청이다
        self.assertEqual((await ai_switch_service.get_status(self.db))["today_analyses"], before + 1)

    async def test_dependency_blocks_when_off(self):
        with patch.object(dependencies, "AsyncSessionLocal", self.sessions):
            await require_ai_enabled()  # 켜져 있으면 통과
            await ai_switch_service.set_ai_enabled(self.db, False)
            with self.assertRaises(HTTPException) as caught:
                await require_ai_enabled()
        self.assertEqual((caught.exception.status_code, caught.exception.detail), (503, MESSAGE))


class _Session:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class GuardedEndpointsTest(unittest.TestCase):
    """스위치가 꺼지면 다섯 엔드포인트 모두 파이프라인을 시작하지 않고 503 + 안내 문구."""

    def setUp(self):
        from main import app

        self.app = app
        self.addCleanup(app.dependency_overrides.clear)
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, is_admin=False)
        for target in (
            patch.object(dependencies, "AsyncSessionLocal", _Session),
            patch.object(ai_switch_service, "is_ai_enabled", AsyncMock(return_value=False)),
        ):
            target.start()
            self.addCleanup(target.stop)

    def test_cost_endpoints_return_503(self):
        client = TestClient(self.app)
        with patch.object(agent_service, "stream_agent") as stream, patch.object(agent_service, "stream_analyze") as analyze, \
                patch.object(agent_service, "extract_pdf_pages") as extract:
            responses = {
                "search": client.post("/api/v1/agent/search", json={"query": "lora"}),
                "analyze": client.post("/api/v1/agent/analyze", json={"paper": {"title": "t"}, "query": "q"}),
                "pdf": client.post("/api/v1/agent/pdf", files={"file": ("p.pdf", b"%PDF-1.4", "application/pdf")}),
                "trend": client.post("/api/v1/agent/trend", json={"topic": "rl"}),
                "ask": client.post("/api/v1/analyses/1/ask", json={"question": "what is this paper about?"}),
            }
        for name, response in responses.items():
            with self.subTest(endpoint=name):
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"], MESSAGE)
        stream.assert_not_called()
        analyze.assert_not_called()
        extract.assert_not_called()  # PDF 파싱(수 초)도 하지 않는다

    def test_read_only_endpoints_stay_open(self):
        # 저장된 기록·추천어 조회는 비용이 들지 않는다 — 꺼도 열려 있다
        response = TestClient(self.app).get("/api/v1/papers/daily-keywords")
        self.assertEqual(response.status_code, 200)


class AdminSwitchApiTest(unittest.TestCase):
    def setUp(self):
        from main import app

        self.app = app
        self.addCleanup(app.dependency_overrides.clear)

        async def _db():
            yield None

        app.dependency_overrides[get_db] = _db
        self.status = {"enabled": False, "updated_at": None, "today_analyses": 3, "today_searches": 5}

    def _as(self, is_admin: bool) -> TestClient:
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, is_admin=is_admin)
        return TestClient(self.app)

    def test_non_admin_cannot_read_or_change(self):
        client = self._as(False)
        with patch.object(ai_switch_service, "set_ai_enabled", AsyncMock()) as change:
            self.assertEqual(client.get("/api/v1/admin/ai-switch").status_code, 403)
            self.assertEqual(client.put("/api/v1/admin/ai-switch", json={"enabled": False}).status_code, 403)
        change.assert_not_called()

    def test_admin_reads_and_changes(self):
        client = self._as(True)
        with patch.object(ai_switch_service, "get_status", AsyncMock(return_value=self.status)), \
                patch.object(ai_switch_service, "set_ai_enabled", AsyncMock(return_value=self.status)) as change:
            self.assertEqual(client.get("/api/v1/admin/ai-switch").json(), self.status)
            response = client.put("/api/v1/admin/ai-switch", json={"enabled": False})
        self.assertEqual(response.json(), self.status)
        self.assertEqual(change.await_args.args[1], False)
        self.assertEqual(client.put("/api/v1/admin/ai-switch", json={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
