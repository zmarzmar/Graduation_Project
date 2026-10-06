"""논문 원문 보관·중복 제거·삭제 규칙 테스트 — 실제 Postgres를 사용한다.

ON CONFLICT, ON DELETE SET NULL 같은 DB 동작은 mock으로 검증할 수 없어서 실제 DB에 붙는다.
모든 테스트는 바깥 트랜잭션 안에서 실행하고 끝나면 롤백한다 — 개발 DB에 흔적을 남기지 않는다.

실행: cd BE && uv run alembic upgrade head && uv run python -m unittest discover -s tests -v
DB에 붙을 수 없으면 건너뛴다. CI에서는 REQUIRE_DB_TESTS=1로 건너뛰기를 실패로 바꾼다.
"""

import asyncio
import os
import unittest
import uuid
from unittest.mock import patch

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from core.config import settings
from crud import analysis as crud_analysis
from crud.paper_document import (
    claim_index_job,
    delete_purged_document,
    fail_index_job,
    finish_index_job,
    get_document_for_analysis,
    get_or_create_document,
    hash_pages,
    list_purge_pending,
    reset_lost_index,
)
from models.analysis import AnalysisResult
from models.paper_document import PaperDocument
from models.user import User
from services import agent_service

PAGES = ["first page", "", "third page"]  # 가운데는 빈 페이지


class PaperDocumentDbTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 테스트마다 이벤트 루프가 새로 생기므로 엔진도 새로 만든다
        self.engine = create_async_engine(settings.database_url, poolclass=NullPool)
        try:
            self.conn = await self.engine.connect()
        except Exception as e:
            await self.engine.dispose()
            if os.environ.get("REQUIRE_DB_TESTS"):
                raise
            self.skipTest(f"database unavailable: {e}")
        self.outer = await self.conn.begin()
        # 안쪽의 commit은 세이브포인트로 처리돼 바깥 트랜잭션을 끝내지 못한다 → 마지막에 전부 롤백
        self.sessions = async_sessionmaker(
            bind=self.conn, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        self.db: AsyncSession = self.sessions()

    async def asyncTearDown(self):
        await self.db.close()
        await self.outer.rollback()
        await self.conn.close()
        await self.engine.dispose()

    async def _user(self) -> int:
        tag = uuid.uuid4().hex[:12]
        user = User(email=f"doc-test-{tag}@example.com", username=f"doc-test-{tag}", password_hash="x")
        self.db.add(user)
        await self.db.flush()
        return user.id

    async def _analysis(self, user_id: int, document_id: int | None) -> AnalysisResult:
        return await crud_analysis.create_analysis_result(
            self.db, mode="pdf", query="paper.pdf", generated_code="", review_feedback="",
            review_passed=True, iteration_count=1, user_id=user_id, document_id=document_id,
        )

    async def _document_ids(self, user_id: int) -> list[int]:
        # 삭제 대기(툼스톤)가 아닌 활성 문서만
        rows = await self.db.execute(
            select(PaperDocument.id).where(PaperDocument.user_id == user_id, PaperDocument.purge_pending_at.is_(None))
        )
        return list(rows.scalars())

    # ── 보관·중복 제거 ────────────────────────────────────────────────────

    async def test_pages_are_stored_with_blank_pages_in_order(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="paper.pdf")
        doc = await self.db.get(PaperDocument, document_id)
        self.assertEqual(doc.pages, PAGES)
        self.assertEqual(doc.page_count, 3)
        self.assertEqual(doc.doc_hash, hash_pages(PAGES))

    async def test_same_user_same_text_reuses_the_document(self):
        user_id = await self._user()
        first = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="a.pdf")
        second = await get_or_create_document(self.db, user_id, list(PAGES), source="upload", title="renamed.pdf")
        self.assertEqual(first, second)
        self.assertEqual(await self._document_ids(user_id), [first])
        # 재사용은 기존 문서의 정보를 덮어쓰지 않는다 — 분석별 정보는 각 분석 기록(query 등)에 남는다
        self.assertEqual((await self.db.get(PaperDocument, first)).title, "a.pdf")

    async def test_hash_is_computed_from_the_pages_actually_stored(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, ["bad\x00text"], source="upload", title="p.pdf")
        doc = await self.db.get(PaperDocument, document_id)
        self.assertEqual(doc.doc_hash, hash_pages(doc.pages))
        # NUL 유무만 다른 본문은 저장되는 내용이 같으므로 같은 문서다
        self.assertEqual(await get_or_create_document(self.db, user_id, ["badtext"], source="upload", title="p.pdf"), document_id)

    async def test_page_boundaries_are_unambiguous_even_with_separator_characters(self):
        self.assertNotEqual(hash_pages(["a\f", "b"]), hash_pages(["a", "\fb"]))
        self.assertNotEqual(hash_pages(['a", "b']), hash_pages(["a", "b"]))

    async def test_documents_are_not_shared_between_users(self):
        alice, bob = await self._user(), await self._user()
        a = await get_or_create_document(self.db, alice, PAGES, source="upload", title="p.pdf")
        b = await get_or_create_document(self.db, bob, PAGES, source="upload", title="p.pdf")
        self.assertNotEqual(a, b)

    async def test_different_page_layout_is_a_different_document(self):
        self.assertNotEqual(hash_pages(["ab", "c"]), hash_pages(["a", "bc"]))

    async def test_nul_characters_do_not_break_storage(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, ["bad\x00text"], source="upload", title="p.pdf")
        self.assertEqual((await self.db.get(PaperDocument, document_id)).pages, ["badtext"])

    # ── 삭제 규칙 ─────────────────────────────────────────────────────────

    async def test_deleting_one_record_keeps_the_document_used_by_another(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        first = await self._analysis(user_id, document_id)
        second = await self._analysis(user_id, document_id)

        self.assertTrue(await crud_analysis.delete_analysis_result_by_id(self.db, first.id, user_id))

        self.assertEqual(await self._document_ids(user_id), [document_id])
        await self.db.refresh(second)
        self.assertEqual(second.document_id, document_id)

    async def test_deleting_the_last_active_record_empties_the_document_and_keeps_a_tombstone(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        first = await self._analysis(user_id, document_id)
        second = await self._analysis(user_id, document_id)

        await crud_analysis.delete_analysis_result_by_id(self.db, first.id, user_id)
        await crud_analysis.delete_analysis_result_by_id(self.db, second.id, user_id)

        self.assertEqual(await self._document_ids(user_id), [])
        tombstone = await self.db.get(PaperDocument, document_id, populate_existing=True)
        # 원문은 즉시 비워지고, 벡터 색인 정리에 필요한 행(id)만 남는다
        self.assertEqual(tombstone.pages, [])
        self.assertIsNotNone(tombstone.purge_pending_at)
        self.assertEqual(tombstone.doc_hash, f"purged:{document_id}")
        self.assertEqual([doc_id for doc_id, _ in await list_purge_pending(self.db)].count(document_id), 1)

    async def test_removing_the_tombstone_keeps_analysis_rows_and_only_unlinks_them(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        record = await self._analysis(user_id, document_id)
        await crud_analysis.delete_analysis_result_by_id(self.db, record.id, user_id)

        await delete_purged_document(self.db, document_id)

        self.assertIsNone(await self.db.get(PaperDocument, document_id, populate_existing=True))
        await self.db.refresh(record)
        self.assertTrue(record.is_deleted)
        self.assertIsNone(record.document_id)

    async def test_an_active_document_is_never_removed_by_the_tombstone_cleanup(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        await self._analysis(user_id, document_id)
        await delete_purged_document(self.db, document_id)  # 툼스톤이 아니므로 아무 일도 없어야 한다
        self.assertEqual(await self._document_ids(user_id), [document_id])

    async def test_reanalysis_while_deletion_is_pending_creates_a_new_document(self):
        user_id = await self._user()
        old_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        record = await self._analysis(user_id, old_id)
        await crud_analysis.delete_analysis_result_by_id(self.db, record.id, user_id)

        new_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")

        # 삭제 중인 문서는 되살리지 않는다 — 새 id라서 옛 문서의 정리 작업이 새 문서를 건드릴 수 없다
        self.assertNotEqual(new_id, old_id)
        self.assertEqual((await self.db.get(PaperDocument, new_id)).pages, PAGES)
        self.assertIsNotNone((await self.db.get(PaperDocument, old_id, populate_existing=True)).purge_pending_at)

    # ── 접근 권한 ─────────────────────────────────────────────────────────

    async def test_document_access_goes_through_the_users_own_active_analysis(self):
        alice, bob = await self._user(), await self._user()
        document_id = await get_or_create_document(self.db, alice, PAGES, source="upload", title="p.pdf")
        record = await self._analysis(alice, document_id)

        self.assertEqual((await get_document_for_analysis(self.db, record.id, alice)).id, document_id)
        self.assertIsNone(await get_document_for_analysis(self.db, record.id, bob))        # 남의 기록
        self.assertIsNone(await get_document_for_analysis(self.db, 10**9, alice))          # 없는 기록
        no_doc = await self._analysis(alice, None)
        self.assertIsNone(await get_document_for_analysis(self.db, no_doc.id, alice))      # 원문 없는 기록

        await crud_analysis.delete_analysis_result_by_id(self.db, record.id, alice)
        self.assertIsNone(await get_document_for_analysis(self.db, record.id, alice))      # 삭제된 기록

    # ── 색인 작업 ─────────────────────────────────────────────────────────

    async def _doc(self) -> int:
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        await self._analysis(user_id, document_id)
        return document_id

    async def _status(self, document_id: int) -> tuple[str, str | None, int | None]:
        doc = await self.db.get(PaperDocument, document_id, populate_existing=True)
        return doc.index_status, doc.index_job_id, doc.indexed_chunk_count

    async def test_only_one_claim_succeeds_until_the_job_ends(self):
        document_id = await self._doc()
        job = await claim_index_job(self.db, document_id, stale_after_seconds=90)
        self.assertIsNotNone(job)
        self.assertIsNone(await claim_index_job(self.db, document_id, stale_after_seconds=90))

        self.assertTrue(await finish_index_job(self.db, document_id, job, chunk_count=7))
        self.assertEqual(await self._status(document_id), ("ready", job, 7))
        self.assertIsNone(await claim_index_job(self.db, document_id, stale_after_seconds=90))  # ready는 다시 색인하지 않는다

    async def test_failed_job_can_be_claimed_again(self):
        document_id = await self._doc()
        job = await claim_index_job(self.db, document_id, stale_after_seconds=90)
        self.assertTrue(await fail_index_job(self.db, document_id, job, "embedding error"))
        retry = await claim_index_job(self.db, document_id, stale_after_seconds=90)
        self.assertIsNotNone(retry)
        self.assertNotEqual(retry, job)

    async def test_a_superseded_job_cannot_finish_or_fail_the_current_one(self):
        document_id = await self._doc()
        old = await claim_index_job(self.db, document_id, stale_after_seconds=90)
        # 오래 걸리는(또는 죽은) 작업으로 보고 다시 선점 — 옛 작업은 아직 돌고 있을 수 있다.
        # 이 테스트는 한 트랜잭션 안에서 돌아 now()가 고정이므로 음수 기준으로 '이미 오래됨'을 만든다.
        new = await claim_index_job(self.db, document_id, stale_after_seconds=-1)
        self.assertIsNotNone(new)

        self.assertFalse(await finish_index_job(self.db, document_id, old, chunk_count=3))
        self.assertFalse(await fail_index_job(self.db, document_id, old, "late failure"))
        self.assertEqual(await self._status(document_id), ("indexing", new, None))

        self.assertTrue(await finish_index_job(self.db, document_id, new, chunk_count=9))
        self.assertFalse(await finish_index_job(self.db, document_id, old, chunk_count=3))  # 완료 뒤에 도착해도 못 바꾼다
        self.assertEqual(await self._status(document_id), ("ready", new, 9))

    async def test_deleting_the_document_invalidates_a_running_index_job(self):
        user_id = await self._user()
        document_id = await get_or_create_document(self.db, user_id, PAGES, source="upload", title="p.pdf")
        record = await self._analysis(user_id, document_id)
        job = await claim_index_job(self.db, document_id, stale_after_seconds=90)

        await crud_analysis.delete_analysis_result_by_id(self.db, record.id, user_id)

        # 색인 중에 삭제됐다 — 늦게 끝난 작업이 완료를 기록하거나 툼스톤을 다시 선점할 수 없다
        self.assertFalse(await finish_index_job(self.db, document_id, job, chunk_count=5))
        self.assertIsNone(await claim_index_job(self.db, document_id, stale_after_seconds=-1))

    async def test_lost_index_is_reset_only_for_the_job_that_was_checked(self):
        document_id = await self._doc()
        job = await claim_index_job(self.db, document_id, stale_after_seconds=90)
        await finish_index_job(self.db, document_id, job, chunk_count=4)

        self.assertFalse(await reset_lost_index(self.db, document_id, "some-other-job"))
        self.assertTrue(await reset_lost_index(self.db, document_id, job))
        self.assertEqual(await self._status(document_id), ("none", None, None))

    async def test_delete_all_removes_only_that_users_documents(self):
        alice, bob = await self._user(), await self._user()
        a = await get_or_create_document(self.db, alice, PAGES, source="upload", title="p.pdf")
        b = await get_or_create_document(self.db, bob, PAGES, source="upload", title="p.pdf")
        await self._analysis(alice, a)
        await self._analysis(bob, b)

        await crud_analysis.delete_all_analysis_results(self.db, alice)

        self.assertEqual(await self._document_ids(alice), [])
        self.assertEqual(await self._document_ids(bob), [b])

    # ── 저장 경로 연결 ────────────────────────────────────────────────────

    async def _save_pdf_run(self, user_id: int | None, pages: list[str]) -> AnalysisResult:
        accumulated = {"pdf_pages": pages, "paper_summary": "요약"}
        with patch.object(agent_service, "AsyncSessionLocal", self.sessions):
            await agent_service._save_to_db("pdf", "paper.pdf", accumulated, user_id=user_id)
        rows = await self.db.execute(
            select(AnalysisResult).where(AnalysisResult.query == "paper.pdf").order_by(AnalysisResult.id.desc())
        )
        return rows.scalars().first()

    async def test_upload_run_links_the_stored_document(self):
        user_id = await self._user()
        record = await self._save_pdf_run(user_id, PAGES)
        doc = await self.db.get(PaperDocument, record.document_id)
        self.assertEqual((doc.user_id, doc.source, doc.title, doc.pages), (user_id, "upload", "paper.pdf", PAGES))

    async def test_download_run_links_the_stored_document(self):
        user_id = await self._user()
        paper = {
            "arxiv_id": f"test.{uuid.uuid4().hex[:8]}", "title": "Attention Is All You Need",
            "authors": ["A. Vaswani"], "abstract": "abs", "url": "https://arxiv.org/abs/1706.03762",
            "pdf_url": "https://arxiv.org/pdf/1706.03762", "published_at": "2017-06-12T00:00:00Z", "categories": ["cs.CL"],
        }
        with patch.object(agent_service, "AsyncSessionLocal", self.sessions):
            await agent_service._save_analyze_to_db("q", paper, {"pdf_pages": PAGES}, user_id=user_id)
        doc = (await self.db.execute(select(PaperDocument).where(PaperDocument.user_id == user_id))).scalar_one()
        self.assertEqual((doc.source, doc.arxiv_id, doc.title), ("arxiv", paper["arxiv_id"], paper["title"]))
        record = (await self.db.execute(select(AnalysisResult).where(AnalysisResult.user_id == user_id))).scalar_one()
        self.assertEqual(record.document_id, doc.id)
        self.assertIsNotNone(record.paper_id)  # 논문 upsert와 원문 보관이 같은 트랜잭션에서 함께 성공한다

    async def test_guest_run_stores_no_document(self):
        before = (await self.db.execute(select(PaperDocument.id))).all()
        record = await self._save_pdf_run(None, PAGES)
        self.assertIsNone(record.document_id)
        self.assertEqual((await self.db.execute(select(PaperDocument.id))).all(), before)

    async def test_abstract_only_run_stores_no_document(self):
        user_id = await self._user()
        record = await self._save_pdf_run(user_id, [])
        self.assertIsNone(record.document_id)
        self.assertEqual(await self._document_ids(user_id), [])


class PaperDocumentConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    """서로 다른 연결에서 동시에 실행되는 저장·삭제. 두 연결이 서로의 데이터를 봐야 하므로 실제로 커밋하고, 끝나면 직접 지운다."""

    async def asyncSetUp(self):
        self.engine = create_async_engine(settings.database_url, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        try:
            async with self.sessions() as db:
                tag = uuid.uuid4().hex[:12]
                user = User(email=f"doc-test-{tag}@example.com", username=f"doc-test-{tag}", password_hash="x")
                db.add(user)
                await db.commit()
                self.user_id = user.id
        except Exception as e:
            await self.engine.dispose()
            if os.environ.get("REQUIRE_DB_TESTS"):
                raise
            self.skipTest(f"database unavailable: {e}")

    async def asyncTearDown(self):
        async with self.sessions() as db:
            # analysis_results.user_id는 SET NULL이라 사용자만 지우면 행이 남는다 — 먼저 지운다
            await db.execute(delete(AnalysisResult).where(AnalysisResult.user_id == self.user_id))
            await db.execute(delete(User).where(User.id == self.user_id))  # paper_documents는 CASCADE
            await db.commit()
        await self.engine.dispose()

    async def _new_analysis(self, db: AsyncSession, document_id: int) -> AnalysisResult:
        return await crud_analysis.create_analysis_result(
            db, mode="pdf", query="paper.pdf", generated_code="", review_feedback="",
            review_passed=True, iteration_count=1, user_id=self.user_id, document_id=document_id,
        )

    async def test_concurrent_saves_of_the_same_text_create_one_document(self):
        async def save() -> int:
            async with self.sessions() as db:
                document_id = await get_or_create_document(db, self.user_id, PAGES, source="upload", title="p.pdf")
                await asyncio.sleep(0.2)  # 트랜잭션을 열어둔 채 겹치게 한다
                await self._new_analysis(db, document_id)
                await db.commit()
                return document_id

        ids = await asyncio.gather(save(), save(), save())

        self.assertEqual(len(set(ids)), 1)
        async with self.sessions() as db:
            count = await db.scalar(
                select(func.count()).select_from(PaperDocument).where(PaperDocument.user_id == self.user_id)
            )
        self.assertEqual(count, 1)

    async def test_deleting_the_last_record_waits_for_a_save_that_is_linking_the_document(self):
        async with self.sessions() as db:
            document_id = await get_or_create_document(db, self.user_id, PAGES, source="upload", title="p.pdf")
            old = await self._new_analysis(db, document_id)
            await db.commit()

        saver = self.sessions()
        try:
            # 저장 쪽: 기존 문서를 찾았지만 아직 새 분석 기록을 커밋하지 않았다
            self.assertEqual(
                await get_or_create_document(saver, self.user_id, PAGES, source="upload", title="p.pdf"), document_id
            )

            async def delete_last_record() -> None:
                async with self.sessions() as db:
                    await crud_analysis.delete_analysis_result_by_id(db, old.id, self.user_id)
                    await db.commit()

            deleter = asyncio.create_task(delete_last_record())
            # 삭제 쪽은 저장이 끝날 때까지 기다려야 한다 — 기다리지 않으면 문서를 지워버린다
            done, _ = await asyncio.wait({deleter}, timeout=0.5)
            self.assertFalse(done, "delete did not wait for the in-flight save")

            new = await self._new_analysis(saver, document_id)
            await saver.commit()
            await deleter
        finally:
            await saver.close()

        async with self.sessions() as db:
            self.assertIsNotNone(await db.get(PaperDocument, document_id))  # 새 기록이 쓰므로 남는다
            self.assertEqual((await db.get(AnalysisResult, new.id)).document_id, document_id)
            self.assertTrue((await db.get(AnalysisResult, old.id)).is_deleted)

    async def test_concurrent_first_questions_claim_the_index_job_exactly_once(self):
        async with self.sessions() as db:
            document_id = await get_or_create_document(db, self.user_id, PAGES, source="upload", title="p.pdf")
            await self._new_analysis(db, document_id)
            await db.commit()

        async def claim() -> str | None:
            async with self.sessions() as db:
                job = await claim_index_job(db, document_id, stale_after_seconds=90)
                await asyncio.sleep(0.2)  # 트랜잭션을 열어둔 채 겹치게 한다
                await db.commit()
                return job

        jobs = await asyncio.gather(*(claim() for _ in range(5)))

        winners = [job for job in jobs if job]
        self.assertEqual(len(winners), 1)
        async with self.sessions() as db:
            self.assertEqual((await db.get(PaperDocument, document_id)).index_job_id, winners[0])


if __name__ == "__main__":
    unittest.main()
