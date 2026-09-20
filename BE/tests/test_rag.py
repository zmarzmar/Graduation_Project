"""논문 Q&A(RAG) 테스트 — 청크 분할, 출처 검증, 색인·삭제의 동시 실행, API, 실제 Chroma 통합.

색인·삭제 경쟁은 실제 Postgres + 가짜 Chroma 컬렉션(메모리)으로 검증한다 — 경쟁의 핵심은 DB 상태 전이와
청크의 job_id 구분이라 Chroma 서버 없이도 확인할 수 있다. 실제 서버와의 호환은 ChromaIntegrationTest가 맡는다.

실행: cd BE && uv run alembic upgrade head && uv run python -m unittest discover -s tests -v
CI에서는 REQUIRE_DB_TESTS=1, REQUIRE_CHROMA_TESTS=1로 건너뛰기를 실패로 바꾼다.
"""

import asyncio
import math
import os
import unittest
import uuid
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import httpx
from sqlalchemy import delete, func, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from agents import qa
from agents.qa import Citation, QaDraft, answer_question, validate_citations
from agents.token_budget import count_tokens
from core.config import settings
from core.dependencies import get_current_user, get_db
from crud import analysis as crud_analysis
from crud.paper_document import claim_index_job, finish_index_job, get_document_for_analysis, get_or_create_document
from models.analysis import AnalysisResult
from models.paper_document import PaperDocument
from models.user import User
from services import rag_service
from services.rag_service import IndexingInProgress, Passage, chunk_pages

# 패치되기 전의 실제 컬렉션 생성 함수 (통합 테스트용)
_REAL_COLLECTION = rag_service._collection

# 서로 다른 주제의 페이지 — 가짜 임베딩이 단어 겹침으로 구분할 수 있게 한다
PAGES = [
    "Low rank adaptation freezes the pretrained weights and injects trainable rank decomposition matrices.",
    "",
    "We evaluate on GLUE with RoBERTa and report accuracy for every task in the benchmark suite.",
]
_VOCAB = ["rank", "adaptation", "weights", "matrices", "glue", "roberta", "accuracy", "benchmark", "bananas"]


def _fake_vector(text: str) -> list[float]:
    words = text.casefold().split()
    return [float(sum(word.startswith(term) for word in words)) + 0.01 for term in _VOCAB]


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_fake_vector(text) for text in texts]


def _matches(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, part) for part in where["$and"])
    [(key, expected)] = where.items()
    if isinstance(expected, dict):
        [(op, value)] = expected.items()
        assert op == "$ne", op
        return meta.get(key) != value
    return meta.get(key) == expected


class FakeCollection:
    """rag_service가 쓰는 만큼의 Chroma 컬렉션 API를 메모리로 흉내 낸다."""

    def __init__(self):
        self.rows: dict[str, tuple[list[float], str, dict]] = {}
        self.fail_deletes = False

    def upsert(self, ids, embeddings, documents, metadatas):
        for row_id, vector, text, meta in zip(ids, embeddings, documents, metadatas):
            self.rows[row_id] = (vector, text, meta)

    def get(self, where=None, include=None):
        return {"ids": [row_id for row_id, (_, _, meta) in self.rows.items() if _matches(meta, where)]}

    def delete(self, where=None):
        if self.fail_deletes:
            raise ConnectionError("chroma unavailable")
        for row_id in self.get(where=where)["ids"]:
            del self.rows[row_id]

    def query(self, query_embeddings, n_results, where=None, include=None):
        [query] = query_embeddings

        def distance(vector: list[float]) -> float:
            dot = sum(a * b for a, b in zip(query, vector))
            return 1 - dot / (math.hypot(*query) * math.hypot(*vector))

        hits = sorted(
            ((distance(vector), text, meta) for vector, text, meta in self.rows.values() if _matches(meta, where)),
            key=lambda hit: hit[0],
        )[:n_results]
        return {
            "documents": [[text for _, text, _ in hits]],
            "metadatas": [[meta for _, _, meta in hits]],
            "distances": [[dist for dist, _, _ in hits]],
        }

    def job_ids(self, document_id: int) -> set[str]:
        return {meta["job_id"] for _, _, meta in self.rows.values() if meta["document_id"] == document_id}


# ── 청크 분할 ─────────────────────────────────────────────────────────────


class ChunkingTest(unittest.TestCase):
    def test_blank_pages_are_skipped_but_page_numbers_are_kept(self):
        chunks = chunk_pages(PAGES)
        self.assertEqual([chunk.page for chunk in chunks], [1, 3])
        self.assertEqual([chunk.index for chunk in chunks], [0, 1])

    def test_a_paragraph_longer_than_the_limit_is_split_at_sentence_boundaries(self):
        sentences = [f"Sentence number {i} explains one step of the method." for i in range(40)]
        chunks = chunk_pages([" ".join(sentences)], limit=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(count_tokens(chunk.text) <= 60 for chunk in chunks))
        self.assertTrue(all(chunk.text.rstrip().endswith(".") for chunk in chunks))  # 문장 중간에서 끊지 않는다
        self.assertEqual(" ".join(chunk.text.replace("\n", " ") for chunk in chunks), " ".join(sentences))

    def test_a_single_sentence_longer_than_the_limit_is_hard_split_with_overlap(self):
        table = " ".join(f"{i * 0.137:.3f}" for i in range(400))  # 문장 경계가 없는 숫자 표
        chunks = chunk_pages([table], limit=80)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(count_tokens(chunk.text) <= 80 for chunk in chunks))
        self.assertTrue(all(chunk.page == 1 for chunk in chunks))
        self.assertIn("54.663", chunks[-1].text)  # 마지막 값까지 빠짐없이 들어간다

    def test_chunking_is_deterministic(self):
        self.assertEqual(chunk_pages(PAGES * 3), chunk_pages(PAGES * 3))


# ── 출처 검증 ─────────────────────────────────────────────────────────────


class CitationValidationTest(unittest.IsolatedAsyncioTestCase):
    PASSAGES = [
        Passage(chunk_index=4, page=2, text="LoRA freezes the pretrained\nmodel weights and injects trainable matrices.", distance=0.1),
        Passage(chunk_index=9, page=5, text="We evaluate on the GLUE benchmark.", distance=0.2),
    ]

    def _draft(self, *citations: tuple[int, str], answerable: bool = True) -> QaDraft:
        return QaDraft(answerable=answerable, answer="LoRA freezes the weights.",
                       citations=[Citation(passage=number, quote=quote) for number, quote in citations])

    def test_only_quotes_that_really_appear_in_the_cited_passage_survive(self):
        valid, dropped = validate_citations(
            self._draft(
                (1, "freezes the pretrained model weights"),       # 줄바꿈·대소문자 차이는 허용
                (1, "LoRA reduces memory by ten thousand times"),  # 그 구절에 없는 문장
                (2, "freezes the pretrained model weights"),       # 다른 구절의 문장
                (7, "We evaluate on the GLUE benchmark."),         # 검색되지 않은 구절 번호
                (2, ""),                                           # 빈 인용문
                (2, "GLUE"),                                       # 너무 짧은 인용문
            ),
            self.PASSAGES,
        )
        self.assertEqual(valid, [{"page": 2, "chunk_index": 4, "quote": "freezes the pretrained model weights"}])
        self.assertEqual(dropped, 5)

    def test_pdf_extraction_artifacts_do_not_make_a_real_quote_look_fake(self):
        # 평가에서 실제로 나온 실패: 추출 텍스트는 "pre-\ntrained"인데 모델은 "pre-trained"로 이어서 인용한다
        passages = [Passage(chunk_index=0, page=1, distance=0.1,
                            text="LoRA, which freezes the pre-\ntrained model weights and injects trainable rank decom-\nposition "
                                 "matrices. The ﬁne-tuned model is efﬁcient for large values of\ndk.")]
        draft = QaDraft(answerable=True, answer="a", citations=[
            Citation(passage=1, quote="freezes the pre-trained model weights and injects trainable rank decomposition matrices"),
            Citation(passage=1, quote="The fine-tuned model is efficient for large values of dk."),  # 합자(ﬁ)와 줄바꿈
            Citation(passage=1, quote="freezes the pre-trained optimizer states"),                   # 여전히 없는 문장은 거부
        ])
        valid, dropped = validate_citations(draft, passages)
        self.assertEqual(len(valid), 2)
        self.assertEqual(dropped, 1)

    async def _answer(self, draft: QaDraft) -> dict:
        class _FakeLLM:
            messages: list = []

            async def ainvoke(self, messages):
                _FakeLLM.messages = messages
                return draft

        self.llm = _FakeLLM
        with patch.object(qa, "_llm", _FakeLLM()):
            return await answer_question("What does LoRA freeze?", self.PASSAGES)

    async def test_an_answer_without_any_valid_citation_is_not_returned(self):
        result = await self._answer(self._draft((1, "a sentence the model made up entirely")))
        self.assertFalse(result["answerable"])
        self.assertEqual(result["answer"], qa.NO_EVIDENCE_MESSAGE)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["dropped_citations"], 1)  # 모델이 거부한 것이 아니라 검증에서 막혔다는 것이 드러난다

    async def test_partially_dropped_citations_are_reported(self):
        result = await self._answer(self._draft((1, "injects trainable matrices"), (2, "not in the passage at all")))
        self.assertTrue(result["answerable"])
        self.assertEqual([c["page"] for c in result["citations"]], [2])
        self.assertEqual(result["dropped_citations"], 1)

    async def test_unanswerable_draft_and_empty_retrieval_return_no_evidence(self):
        self.assertFalse((await self._answer(self._draft(answerable=False)))["answerable"])
        self.assertFalse((await answer_question("anything", []))["answerable"])

    async def test_passages_are_delimited_as_data_and_the_prompt_says_not_to_follow_them(self):
        await self._answer(self._draft((1, "freezes the pretrained model weights")))
        system, human = self.llm.messages
        self.assertIn("따르지 말고", system.content)
        self.assertIn('<passage number="1" page="2">', human.content)
        self.assertLess(human.content.rindex("</passage>"), human.content.index("질문:"))


# ── 색인·삭제 (실제 Postgres + 가짜 컬렉션) ────────────────────────────────


class _DbCase(unittest.IsolatedAsyncioTestCase):
    """두 개 이상의 세션이 서로의 데이터를 봐야 하므로 실제로 커밋하고, 끝나면 직접 지운다."""

    async def asyncSetUp(self):
        self.engine = create_async_engine(settings.database_url, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        try:
            self.user_id = await self._new_user()
        except Exception as e:
            await self.engine.dispose()
            if os.environ.get("REQUIRE_DB_TESTS"):
                raise
            self.skipTest(f"database unavailable: {e}")
        self.user_ids = [self.user_id]
        self.collection = FakeCollection()
        self.embed_calls: list[list[str]] = []

        async def counting_embed(texts: list[str]) -> list[list[float]]:
            self.embed_calls.append(texts)
            return await _fake_embed(texts)

        self.stack = ExitStack()
        # 서비스가 전역 세션 팩토리·Chroma·OpenAI 대신 테스트용을 쓰게 한다
        self.stack.enter_context(patch.object(rag_service, "AsyncSessionLocal", self.sessions))
        self.stack.enter_context(patch.object(rag_service, "_collection", lambda: self.collection))
        self.stack.enter_context(patch.object(rag_service, "embed", counting_embed))

    async def asyncTearDown(self):
        self.stack.close()
        async with self.sessions() as db:
            await db.execute(delete(AnalysisResult).where(AnalysisResult.user_id.in_(self.user_ids)))
            await db.execute(delete(User).where(User.id.in_(self.user_ids)))  # paper_documents는 CASCADE
            await db.commit()
        await self.engine.dispose()

    async def _new_user(self) -> int:
        async with self.sessions() as db:
            tag = uuid.uuid4().hex[:12]
            user = User(email=f"doc-test-{tag}@example.com", username=f"doc-test-{tag}", password_hash="x")
            db.add(user)
            await db.commit()
            return user.id

    async def _analyzed_document(self, user_id: int | None = None, pages: list[str] | None = None) -> tuple[int, int]:
        """(analysis_id, document_id)"""
        user_id = user_id or self.user_id
        async with self.sessions() as db:
            document_id = await get_or_create_document(db, user_id, pages or PAGES, source="upload", title="p.pdf")
            record = await crud_analysis.create_analysis_result(
                db, mode="pdf", query="p.pdf", generated_code="", review_feedback="", review_passed=True,
                iteration_count=1, user_id=user_id, document_id=document_id,
            )
            await db.commit()
            return record.id, document_id

    async def _document(self, document_id: int) -> PaperDocument | None:
        async with self.sessions() as db:
            return await db.get(PaperDocument, document_id)

    async def _delete_record(self, analysis_id: int) -> None:
        async with self.sessions() as db:
            await crud_analysis.delete_analysis_result_by_id(db, analysis_id, self.user_id)
            await db.commit()


class IndexingTest(_DbCase):
    async def test_first_question_indexes_the_document_and_retrieval_stays_inside_it(self):
        _, doc_id = await self._analyzed_document()
        other_user = await self._new_user()
        self.user_ids.append(other_user)
        _, other_doc_id = await self._analyzed_document(other_user, ["Bananas are rich in potassium and bananas are yellow."])

        job = await rag_service.ensure_indexed(await self._document(doc_id))
        await rag_service.ensure_indexed(await self._document(other_doc_id))

        document = await self._document(doc_id)
        self.assertEqual((document.index_status, document.index_job_id, document.indexed_chunk_count), ("ready", job, 2))
        passages = await rag_service.retrieve(document, job, "glue benchmark accuracy with roberta")
        self.assertEqual(passages[0].page, 3)
        # 다른 사용자의 문서에만 있는 내용을 물어도 그 문서의 청크는 절대 나오지 않는다
        about_bananas = await rag_service.retrieve(document, job, "bananas")
        self.assertTrue(all("anana" not in passage.text for passage in about_bananas))
        self.assertEqual({passage.page for passage in about_bananas}, {1, 3})

    async def test_ready_document_is_not_indexed_again(self):
        _, doc_id = await self._analyzed_document()
        first = await rag_service.ensure_indexed(await self._document(doc_id))
        second = await rag_service.ensure_indexed(await self._document(doc_id))
        self.assertEqual(first, second)
        self.assertEqual(len(self.embed_calls), 1)

    async def test_concurrent_first_questions_embed_the_document_only_once(self):
        _, doc_id = await self._analyzed_document()
        document = await self._document(doc_id)  # 모두 'none' 상태를 보고 동시에 시작한다

        results = await asyncio.gather(*(rag_service.ensure_indexed(document) for _ in range(4)), return_exceptions=True)

        self.assertEqual(sum(isinstance(result, str) for result in results), 1)
        self.assertEqual(sum(isinstance(result, IndexingInProgress) for result in results), 3)
        self.assertEqual(len(self.embed_calls), 1)  # upsert가 아니라 선점이 중복 임베딩을 막는다

    async def test_a_slow_superseded_job_cannot_damage_the_new_index(self):
        _, doc_id = await self._analyzed_document()
        # 옛 작업: 선점하고 청크 일부를 쓴 채 느리게 돌고 있다
        async with self.sessions() as db:
            old_job = await claim_index_job(db, doc_id, stale_after_seconds=90)
            await db.execute(
                update(PaperDocument).where(PaperDocument.id == doc_id)
                .values(index_started_at=func.now() - timedelta(minutes=10))
            )
            await db.commit()

        def old_job_writes_a_chunk():
            self.collection.upsert(
                ids=[f"{doc_id}:{old_job}:0"], embeddings=[_fake_vector("glue roberta accuracy benchmark")],
                documents=["STALE CHUNK from the old job"],
                metadatas=[{"document_id": doc_id, "job_id": old_job, "page": 99, "chunk_index": 0}],
            )

        old_job_writes_a_chunk()

        new_job = await rag_service.ensure_indexed(await self._document(doc_id))  # 오래된 작업으로 보고 다시 선점
        self.assertNotEqual(new_job, old_job)

        # 옛 작업이 뒤늦게 청크를 더 쓰고 완료·실패를 기록하려 한다
        old_job_writes_a_chunk()
        async with self.sessions() as db:
            self.assertFalse(await finish_index_job(db, doc_id, old_job, chunk_count=1))
            await db.commit()
        await rag_service._abandon_job(doc_id, old_job, "superseded")

        document = await self._document(doc_id)
        self.assertEqual((document.index_status, document.index_job_id), ("ready", new_job))  # failed로 바뀌지 않았다
        self.assertEqual(self.collection.job_ids(doc_id), {new_job})                          # 자기 청크만 지웠다
        passages = await rag_service.retrieve(document, new_job, "glue roberta accuracy benchmark")
        self.assertNotIn("STALE", " ".join(passage.text for passage in passages))

    async def test_stale_chunks_written_after_the_new_index_is_ready_are_never_searched(self):
        _, doc_id = await self._analyzed_document()
        job = await rag_service.ensure_indexed(await self._document(doc_id))
        self.collection.upsert(
            ids=[f"{doc_id}:deadjob:0"], embeddings=[_fake_vector("glue roberta accuracy benchmark")],
            documents=["STALE CHUNK"], metadatas=[{"document_id": doc_id, "job_id": "deadjob", "page": 99, "chunk_index": 0}],
        )
        passages = await rag_service.retrieve(await self._document(doc_id), job, "glue roberta accuracy benchmark")
        self.assertNotIn("STALE CHUNK", [passage.text for passage in passages])

    async def test_failed_indexing_is_recorded_cleaned_up_and_can_be_retried(self):
        _, doc_id = await self._analyzed_document()

        async def broken_embed(texts):
            raise RuntimeError("embedding API down")

        with patch.object(rag_service, "embed", broken_embed), self.assertRaises(RuntimeError):
            await rag_service.ensure_indexed(await self._document(doc_id))
        document = await self._document(doc_id)
        self.assertEqual(document.index_status, "failed")
        self.assertIn("embedding API down", document.index_error)
        self.assertEqual(self.collection.rows, {})

        await rag_service.ensure_indexed(document)
        self.assertEqual((await self._document(doc_id)).index_status, "ready")

    async def test_indexing_that_exceeds_the_time_limit_fails_instead_of_hanging(self):
        _, doc_id = await self._analyzed_document()

        async def slow_embed(texts):
            await asyncio.sleep(5)
            return await _fake_embed(texts)

        with (
            patch.object(rag_service, "embed", slow_embed),
            patch.object(rag_service, "INDEX_TIME_LIMIT_SECONDS", 0.2),
            self.assertRaises(TimeoutError),
        ):
            await rag_service.ensure_indexed(await self._document(doc_id))
        self.assertEqual((await self._document(doc_id)).index_status, "failed")

    async def test_cancelled_request_releases_the_job(self):
        _, doc_id = await self._analyzed_document()
        started = asyncio.Event()

        async def hanging_embed(texts):
            started.set()
            await asyncio.sleep(30)

        with patch.object(rag_service, "embed", hanging_embed):
            task = asyncio.create_task(rag_service.ensure_indexed(await self._document(doc_id)))
            await started.wait()
            task.cancel()  # 클라이언트가 연결을 끊었다
            with self.assertRaises(asyncio.CancelledError):
                await task
        # 오래된 'indexing'으로 남지 않고 바로 다시 시도할 수 있다
        self.assertEqual((await self._document(doc_id)).index_status, "failed")


class LostIndexTest(_DbCase):
    async def test_no_close_passage_is_a_normal_no_evidence_answer_not_a_lost_index(self):
        _, doc_id = await self._analyzed_document()
        await rag_service.ensure_indexed(await self._document(doc_id))

        with patch.object(settings, "qa_max_distance", 0.05):
            result = await rag_service.ask(await self._document(doc_id), "bananas bananas bananas")

        self.assertFalse(result["answerable"])
        self.assertEqual((await self._document(doc_id)).index_status, "ready")  # 근거 부족으로 재색인하지 않는다
        self.assertEqual(len(self.embed_calls), 2)                              # 색인 1회 + 질문 1회뿐

    async def test_missing_chunks_of_a_ready_document_trigger_a_rebuild(self):
        _, doc_id = await self._analyzed_document()
        job = await rag_service.ensure_indexed(await self._document(doc_id))
        self.collection.rows.clear()  # Chroma 데이터가 사라졌다

        with self.assertRaises(IndexingInProgress):
            await rag_service.retrieve(await self._document(doc_id), job, "glue")
        self.assertEqual((await self._document(doc_id)).index_status, "none")

        rebuilt = await rag_service.ensure_indexed(await self._document(doc_id))  # 다음 요청이 다시 만든다
        self.assertNotEqual(rebuilt, job)
        self.assertTrue(await rag_service.retrieve(await self._document(doc_id), rebuilt, "glue"))


class DeletionTest(_DbCase):
    async def test_deleting_the_last_record_blocks_access_at_once_and_removes_chunks(self):
        analysis_id, doc_id = await self._analyzed_document()
        await rag_service.ensure_indexed(await self._document(doc_id))

        await self._delete_record(analysis_id)
        await rag_service.purge_pending_documents()

        self.assertEqual(self.collection.rows, {})
        tombstone = await self._document(doc_id)
        self.assertEqual(tombstone.pages, [])  # 원문은 이미 없다
        # 유예 시간이 지나기 전에는 행을 남긴다 — 늦게 도착할 수 있는 upsert를 한 번 더 지우기 위해
        self.assertIsNotNone(tombstone.purge_pending_at)

        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()
        self.assertIsNone(await self._document(doc_id))

    async def test_chroma_failure_keeps_the_tombstone_for_retry_and_never_restores_access(self):
        analysis_id, doc_id = await self._analyzed_document()
        await rag_service.ensure_indexed(await self._document(doc_id))
        await self._delete_record(analysis_id)

        self.collection.fail_deletes = True
        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()  # 예외를 밖으로 던지지 않는다

        tombstone = await self._document(doc_id)
        self.assertIsNotNone(tombstone)                 # 재시도에 필요한 id가 DB에 남아 있다
        self.assertEqual(tombstone.pages, [])
        self.assertTrue(self.collection.rows)           # 청크는 아직 남아 있다 — '완전히 삭제됨'이 아니다
        async with self.sessions() as db:
            self.assertIsNone(await get_document_for_analysis(db, analysis_id, self.user_id))

        self.collection.fail_deletes = False
        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()  # 다음 시도
        self.assertEqual(self.collection.rows, {})
        self.assertIsNone(await self._document(doc_id))

    async def test_document_deleted_while_being_indexed_leaves_no_chunks_behind(self):
        analysis_id, doc_id = await self._analyzed_document()
        document = await self._document(doc_id)

        async def embed_while_the_user_deletes(texts):
            # 색인 도중에 마지막 기록이 삭제되고 정리까지 한 번 돈다
            await self._delete_record(analysis_id)
            await rag_service.purge_pending_documents()
            return await _fake_embed(texts)

        with patch.object(rag_service, "embed", embed_while_the_user_deletes), self.assertRaises(IndexingInProgress):
            await rag_service.ensure_indexed(document)  # 정리 뒤에 upsert가 도착하지만 완료는 기록할 수 없다

        self.assertEqual(self.collection.rows, {})  # 늦게 쓴 청크는 작업 스스로 지웠다
        tombstone = await self._document(doc_id)
        self.assertEqual((tombstone.index_status, tombstone.index_job_id), ("none", None))

        # 작업이 자기 청크를 못 지우고 죽었더라도, 유예 시간 뒤의 정리가 한 번 더 지우고 나서 행을 없앤다
        self.collection.upsert(ids=[f"{doc_id}:crashed:0"], embeddings=[_fake_vector("rank")], documents=["late"],
                               metadatas=[{"document_id": doc_id, "job_id": "crashed", "page": 1, "chunk_index": 0}])
        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()
        self.assertEqual(self.collection.rows, {})
        self.assertIsNone(await self._document(doc_id))

    async def test_reanalysis_during_cleanup_gets_a_new_document_the_cleanup_cannot_touch(self):
        analysis_id, old_id = await self._analyzed_document()
        await rag_service.ensure_indexed(await self._document(old_id))
        await self._delete_record(analysis_id)

        # 정리가 끝나기 전에 같은 논문을 다시 분석하고 질문까지 한다
        _, new_id = await self._analyzed_document()
        self.assertNotEqual(new_id, old_id)
        new_job = await rag_service.ensure_indexed(await self._document(new_id))

        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()  # 옛 문서의 정리가 이제야 돈다

        self.assertIsNone(await self._document(old_id))
        new_document = await self._document(new_id)
        self.assertEqual((new_document.index_status, new_document.pages), ("ready", PAGES))
        self.assertEqual(self.collection.job_ids(new_id), {new_job})
        self.assertTrue(await rag_service.retrieve(new_document, new_job, "glue"))


# ── API ──────────────────────────────────────────────────────────────────


class AskApiTest(_DbCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        from main import app

        async def override_db():
            async with self.sessions() as session:
                yield session
                await session.commit()

        self.app = app
        app.dependency_overrides[get_db] = override_db
        self.current_user: int | None = self.user_id
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=self.current_user)
        self.addCleanup(app.dependency_overrides.clear)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

        class _FakeLLM:
            async def ainvoke(self, messages):
                return QaDraft(answerable=True, answer="GLUE로 평가했다.",
                               citations=[Citation(passage=1, quote="We evaluate on GLUE with RoBERTa")])

        self.stack.enter_context(patch.object(qa, "_llm", _FakeLLM()))

    async def _ask(self, analysis_id: int, question: str = "glue roberta accuracy benchmark") -> httpx.Response:
        return await self.client.post(f"/api/v1/analyses/{analysis_id}/ask", json={"question": question})

    async def test_answer_comes_with_verified_page_citations(self):
        analysis_id, _ = await self._analyzed_document()
        response = await self._ask(analysis_id)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["answerable"])
        self.assertEqual(body["citations"], [{"page": 3, "chunk_index": 1, "quote": "We evaluate on GLUE with RoBERTa"}])

    async def test_guests_are_rejected(self):
        analysis_id, _ = await self._analyzed_document()
        self.app.dependency_overrides.pop(get_current_user)  # 실제 인증 — 토큰 없음
        self.assertEqual((await self._ask(analysis_id)).status_code, 401)

    async def test_other_users_and_deleted_records_are_not_found(self):
        analysis_id, _ = await self._analyzed_document()
        other_user = await self._new_user()
        self.user_ids.append(other_user)

        self.current_user = other_user
        self.assertEqual((await self._ask(analysis_id)).status_code, 404)

        self.current_user = self.user_id
        await self._delete_record(analysis_id)
        self.assertEqual((await self._ask(analysis_id)).status_code, 404)

    async def test_record_without_a_stored_document_explains_that_reanalysis_is_needed(self):
        async with self.sessions() as db:
            record = await crud_analysis.create_analysis_result(
                db, mode="pdf", query="old.pdf", generated_code="", review_feedback="", review_passed=True,
                iteration_count=1, user_id=self.user_id, document_id=None,
            )
            await db.commit()
        response = await self._ask(record.id)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["reason"], "no_document")
        self.assertIn("다시 분석", response.json()["detail"])

    async def test_request_during_indexing_gets_202_with_a_retry_interval(self):
        analysis_id, doc_id = await self._analyzed_document()
        async with self.sessions() as db:
            await claim_index_job(db, doc_id, stale_after_seconds=90)  # 다른 요청이 색인 중
            await db.commit()
        response = await self._ask(analysis_id)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.headers["retry-after"], str(rag_service.RETRY_AFTER_SECONDS))
        self.assertEqual(response.json()["retry_after_seconds"], rag_service.RETRY_AFTER_SECONDS)
        self.assertEqual(self.embed_calls, [])

    async def test_question_length_is_limited(self):
        analysis_id, _ = await self._analyzed_document()
        self.assertEqual((await self._ask(analysis_id, "x" * 501)).status_code, 422)


# ── 실제 Chroma 서버 ─────────────────────────────────────────────────────


class ChromaIntegrationTest(_DbCase):
    """고정한 서버 이미지·클라이언트 버전 조합에서 색인 → 범위 검색 → 삭제가 실제로 동작하는지 확인한다.

    가짜 임베딩은 9차원이라 운영 컬렉션(paper_chunks)을 쓰면 차원이 굳어버린다 — 테스트 전용 컬렉션을 만들고 끝나면 지운다.
    """

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.collection_name = f"paper_chunks_test_{uuid.uuid4().hex[:12]}"
        self.stack.enter_context(patch.object(rag_service, "_COLLECTION", self.collection_name))
        _REAL_COLLECTION.cache_clear()
        try:
            self.collection = await asyncio.to_thread(_REAL_COLLECTION)  # _DbCase의 패치가 이 값을 읽는다
        except Exception as e:
            _REAL_COLLECTION.cache_clear()
            if os.environ.get("REQUIRE_CHROMA_TESTS"):
                raise
            self.skipTest(f"chroma unavailable: {e}")

    async def asyncTearDown(self):
        _REAL_COLLECTION.cache_clear()  # 테스트 컬렉션이 캐시에 남지 않게 한다
        if isinstance(getattr(self, "collection", None), FakeCollection) is False and hasattr(self, "collection_name"):
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            await asyncio.to_thread(client.delete_collection, self.collection_name)
        await super().asyncTearDown()

    async def test_index_scoped_search_and_purge_against_the_real_server(self):
        analysis_id, doc_id = await self._analyzed_document()
        other_user = await self._new_user()
        self.user_ids.append(other_user)
        _, other_id = await self._analyzed_document(other_user, ["Bananas are rich in potassium and bananas are yellow."])

        job = await rag_service.ensure_indexed(await self._document(doc_id))
        other_job = await rag_service.ensure_indexed(await self._document(other_id))
        self.assertEqual(await rag_service._count_chunks(doc_id, job), 2)

        document = await self._document(doc_id)
        self.assertEqual((await rag_service.retrieve(document, job, "glue roberta accuracy benchmark"))[0].page, 3)
        self.assertTrue(all("anana" not in p.text for p in await rag_service.retrieve(document, job, "bananas")))

        await self._delete_record(analysis_id)
        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()
        self.assertEqual(await rag_service._count_chunks(doc_id, job), 0)
        self.assertEqual(await rag_service._count_chunks(other_id, other_job), 1)  # 다른 문서는 그대로
        self.assertIsNone(await self._document(doc_id))


if __name__ == "__main__":
    unittest.main()
