"""논문 Q&A(RAG) 테스트 — 청크 분할, 출처 검증, 색인·삭제의 동시 실행, API, 실제 Chroma 통합.

색인·삭제 경쟁은 실제 Postgres + 가짜 Chroma 컬렉션(메모리)으로 검증한다 — 경쟁의 핵심은 DB 상태 전이와
청크의 job_id 구분이라 Chroma 서버 없이도 확인할 수 있다. 실제 서버와의 호환은 ChromaIntegrationTest가 맡는다.

실행: cd BE && uv run alembic upgrade head && uv run python -m unittest discover -s tests -v
CI에서는 REQUIRE_DB_TESTS=1, REQUIRE_CHROMA_TESTS=1로 건너뛰기를 실패로 바꾼다.
"""

import asyncio
import math
import os
import socket
import time
import unittest
import uuid
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import httpx
import uvicorn
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from agents import qa
from agents.qa import Citation, Claim, QaDraft, answer_question, validate_citations
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

    def get(self, where=None, include=None, limit=None, offset=0):
        matched = [(row_id, meta) for row_id, (_, _, meta) in self.rows.items() if _matches(meta, where)]
        matched = matched[offset:offset + limit] if limit is not None else matched[offset:]
        return {"ids": [row_id for row_id, _ in matched], "metadatas": [meta for _, meta in matched]}

    def delete(self, where=None, ids=None):
        if self.fail_deletes:
            raise ConnectionError("chroma unavailable")
        for row_id in (ids if ids is not None else self.get(where=where)["ids"]):
            self.rows.pop(row_id, None)

    def query(self, query_embeddings, n_results, where=None, include=None):
        def hits_for(query: list[float]) -> list[tuple[float, str, dict]]:
            def distance(vector: list[float]) -> float:
                dot = sum(a * b for a, b in zip(query, vector))
                return 1 - dot / (math.hypot(*query) * math.hypot(*vector))

            return sorted(
                ((distance(vector), text, meta) for vector, text, meta in self.rows.values() if _matches(meta, where)),
                key=lambda hit: hit[0],
            )[:n_results]

        results = [hits_for(query) for query in query_embeddings]
        return {
            "documents": [[text for _, text, _ in hits] for hits in results],
            "metadatas": [[meta for _, _, meta in hits] for hits in results],
            "distances": [[dist for dist, _, _ in hits] for hits in results],
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

    @staticmethod
    def _citations(*citations: tuple[int, str]) -> list[Citation]:
        return [Citation(passage=number, quote=quote) for number, quote in citations]

    def _draft(self, *claims: tuple[str, list[tuple[int, str]]]) -> QaDraft:
        return QaDraft(answerable=True, claims=[Claim(text=text, citations=self._citations(*cited)) for text, cited in claims])

    def test_only_quotes_that_really_appear_in_the_cited_passage_survive(self):
        valid, dropped = validate_citations(
            self._citations(
                (1, "freezes the pretrained model weights"),       # 줄바꿈 차이는 허용
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
        valid, dropped = validate_citations(self._citations(
            (1, "freezes the pre-trained model weights and injects trainable rank decomposition matrices"),
            (1, "freezes the pretrained model weights"),                      # 줄 끝 하이픈은 원래 있었는지 알 수 없다 — 둘 다 허용
            (1, "The fine-tuned model is efficient for large values of dk."),  # 합자(ﬁ)와 줄바꿈
            (1, "freezes the pre-trained optimizer states"),                   # 여전히 없는 문장은 거부
        ), passages)
        self.assertEqual(len(valid), 3)
        self.assertEqual(dropped, 1)

    def test_case_superscripts_and_inline_hyphens_are_compared_as_written(self):
        passage = Passage(chunk_index=0, page=1, distance=0.1,
                          text="The matrix A is multiplied by the scalar a here. The loss grows with x² over time. "
                               "We call it the alpha-beta schedule here. The betagamma variant is used later. "
                               "The gap is x-\ny for the pair.")

        def accepted(quote: str) -> bool:
            return bool(validate_citations(self._citations((1, quote)), [passage])[0])

        # 원문 그대로는 통과
        self.assertTrue(accepted("The matrix A is multiplied by the scalar a here."))
        self.assertTrue(accepted("The loss grows with x² over time."))
        self.assertTrue(accepted("We call it the alpha-beta schedule here."))
        self.assertTrue(accepted("The gap is x-y for the pair."))
        # 대소문자: 행렬 A와 스칼라 a는 다른 기호다
        self.assertFalse(accepted("The matrix a is multiplied by the scalar a here."))
        self.assertFalse(accepted("The matrix A is multiplied by the scalar A here."))
        # 위첨자: x²와 x2는 다르다
        self.assertFalse(accepted("The loss grows with x2 over time."))
        # 줄 안의 하이픈은 넣지도 빼지도 못한다
        self.assertFalse(accepted("We call it the alphabeta schedule here."))
        self.assertFalse(accepted("The beta-gamma variant is used later."))
        # 줄 끝의 하이픈이라도 한 글자 변수 사이면 뺄셈일 수 있다 — 빼면 다른 식이다
        self.assertFalse(accepted("The gap is xy for the pair."))

    def test_whitespace_may_differ_in_amount_but_not_in_presence(self):
        passage = Passage(chunk_index=0, page=1, distance=0.1,
                          text="The product of x y and the\nvalue  xy differ here. The loss is a - b for the pair. "
                               "The update is applied when x > 0 holds.")

        def accepted(quote: str) -> bool:
            return bool(validate_citations(self._citations((1, quote)), [passage])[0])

        # 줄바꿈과 연속 공백은 공백 하나와 같다
        self.assertTrue(accepted("The product of x y and the value xy differ here."))
        self.assertTrue(accepted("The  product of x y\nand the value xy differ here."))
        # 공백을 없애거나 새로 넣으면 다른 문장이다: "x y"(두 변수)와 "xy"(곱 또는 한 변수)
        self.assertFalse(accepted("The product of xy and the value xy differ here."))
        self.assertFalse(accepted("The product of x y and the value x y differ here."))
        self.assertFalse(accepted("The loss is a -b for the pair."))   # 뺄셈이 음수 부호가 된다
        self.assertFalse(accepted("The loss is a-b for the pair."))
        self.assertFalse(accepted("applied when x>0 holds"))           # 뜻은 같아도 원문과 다르다 — 알려진 한계로 둔다

    def test_symbols_that_change_the_meaning_are_never_normalized_away(self):
        passage = Passage(chunk_index=0, page=1, distance=0.1,
                          text="The update is applied when x > 0 holds. The offset is set to -1 in this case. "
                               "We use a dropout rate of 0.1 throughout. The loss is a - b for the pair.")

        def accepted(quote: str) -> bool:
            return bool(validate_citations(self._citations((1, quote)), [passage])[0])

        # 원문 그대로는 통과 (공백의 양만 다른 것은 허용)
        self.assertTrue(accepted("applied when x  >  0 holds"))
        self.assertTrue(accepted("The offset is set to -1 in this case."))
        self.assertTrue(accepted("We use a dropout rate of 0.1 throughout."))
        # 부호·부등호·소수점·뺄셈 기호가 다르면 다른 문장이다
        self.assertFalse(accepted("applied when x < 0 holds"))
        self.assertFalse(accepted("The offset is set to 1 in this case."))
        self.assertFalse(accepted("We use a dropout rate of 01 throughout."))
        self.assertFalse(accepted("The loss is ab for the pair."))
        # 모델이 문장 중간에서 인용을 끊고 마침표로 닫는 경우만 허용한다 (평가에서 실제로 나온 경우) — 안쪽 기호는 그대로 비교
        self.assertTrue(accepted("The update is applied when x > 0 holds. The offset is set to -1."))   # 원문은 "-1 in this case"
        self.assertFalse(accepted("The update is applied when x > 0 holds, the offset is set to -1."))  # 안쪽의 '.' → ','
        self.assertFalse(accepted("We use a dropout rate of 0/1 throughout."))                          # 없는 기호를 넣었다

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
        result = await self._answer(self._draft(("LoRA is fast.", [(1, "a sentence the model made up entirely")])))
        self.assertFalse(result["answerable"])
        self.assertEqual(result["answer"], qa.NO_EVIDENCE_MESSAGE)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["dropped_citations"], 1)  # 모델이 거부한 것이 아니라 검증에서 막혔다는 것이 드러난다
        self.assertEqual(result["dropped_claims"], 1)

    async def test_sentences_whose_citations_fail_are_removed_from_the_answer(self):
        result = await self._answer(self._draft(
            ("LoRA injects trainable matrices.", [(1, "injects trainable matrices")]),
            # 출처 하나는 진짜지만 다른 하나가 가짜다 — 가짜가 받치던 내용이 남지 않게 문장째로 뺀다
            ("It cuts memory use by 10,000 times.", [(2, "We evaluate on the GLUE benchmark."), (2, "not in the passage at all")]),
            ("It was trained on Mars.", []),  # 출처가 없는 문장
            ("LoRA is evaluated on GLUE.", [(2, "We evaluate on the GLUE benchmark."), (2, "We evaluate on the GLUE benchmark.")]),
        ))
        self.assertTrue(result["answerable"])
        self.assertEqual(result["answer"], "LoRA injects trainable matrices. LoRA is evaluated on GLUE.")
        # 빠진 문장의 출처는 싣지 않고, 같은 출처는 한 번만 싣는다
        self.assertEqual([(c["page"], c["quote"]) for c in result["citations"]],
                         [(2, "injects trainable matrices"), (5, "We evaluate on the GLUE benchmark.")])
        self.assertEqual((result["dropped_citations"], result["dropped_claims"]), (1, 2))

    async def test_unanswerable_draft_and_empty_retrieval_return_no_evidence(self):
        self.assertFalse((await self._answer(QaDraft(answerable=False)))["answerable"])
        self.assertFalse((await answer_question("anything", []))["answerable"])

    async def test_passages_are_delimited_as_data_and_the_prompt_says_not_to_follow_them(self):
        await self._answer(self._draft(("LoRA freezes the weights.", [(1, "freezes the pretrained model weights")])))
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

    async def test_cancelled_task_releases_the_job(self):
        _, doc_id = await self._analyzed_document()
        started = asyncio.Event()

        async def hanging_embed(texts):
            started.set()
            await asyncio.sleep(30)

        with patch.object(rag_service, "embed", hanging_embed):
            task = asyncio.create_task(rag_service.ensure_indexed(await self._document(doc_id)))
            await started.wait()
            task.cancel()  # 서버 종료 등으로 태스크가 취소됐다 (연결 종료는 ClientDisconnectTest 참고)
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


class OrphanChunkTest(_DbCase):
    """유예 시간에 기대지 않는 정리 — 요청을 취소해도 Chroma가 이미 받은 쓰기까지 취소된다는 보장은 없다."""

    def _late_write(self, document_id: int, job_id: str, text: str = "late write") -> str:
        chunk_id = f"{document_id}:{job_id}:0"
        self.collection.upsert(ids=[chunk_id], embeddings=[_fake_vector("rank")], documents=[text],
                               metadatas=[{"document_id": document_id, "job_id": job_id, "page": 1, "chunk_index": 0}])
        return chunk_id

    async def test_write_arriving_after_the_tombstone_is_gone_is_found_and_removed(self):
        analysis_id, doc_id = await self._analyzed_document()
        _, kept_id = await self._analyzed_document(pages=["Bananas are rich in potassium and bananas are yellow."])
        job = await rag_service.ensure_indexed(await self._document(doc_id))
        kept_job = await rag_service.ensure_indexed(await self._document(kept_id))

        await self._delete_record(analysis_id)
        with patch.object(rag_service, "PURGE_GRACE_SECONDS", 0):
            await rag_service.purge_pending_documents()
        self.assertIsNone(await self._document(doc_id))  # 툼스톤까지 사라졌다 — DB에는 이 문서의 단서가 없다

        late = self._late_write(doc_id, job)              # 그 뒤에야 Chroma에 도착한 쓰기
        self.assertEqual(await rag_service.reconcile_orphan_chunks(), 1)
        self.assertNotIn(late, self.collection.rows)
        self.assertEqual(self.collection.job_ids(kept_id), {kept_job})  # 살아 있는 문서의 색인은 그대로

    async def test_chunks_of_superseded_jobs_are_removed_but_the_current_index_is_kept(self):
        _, doc_id = await self._analyzed_document()
        job = await rag_service.ensure_indexed(await self._document(doc_id))
        stale = self._late_write(doc_id, "deadjob")
        self.assertEqual(await rag_service.reconcile_orphan_chunks(), 1)
        self.assertNotIn(stale, self.collection.rows)
        self.assertEqual(await rag_service._count_chunks(doc_id, job), 2)
        self.assertEqual(await rag_service.reconcile_orphan_chunks(), 0)  # 다시 돌려도 지울 것이 없다

    async def test_chunks_of_a_job_that_is_still_indexing_are_not_treated_as_orphans(self):
        _, doc_id = await self._analyzed_document()
        async with self.sessions() as db:
            running = await claim_index_job(db, doc_id, stale_after_seconds=90)
            await db.commit()
        partial = self._late_write(doc_id, running, "first chunk of a running job")
        self.assertEqual(await rag_service.reconcile_orphan_chunks(), 0)
        self.assertIn(partial, self.collection.rows)

    async def test_periodic_purge_removes_late_orphans_and_survives_a_failed_round(self):
        self.collection.fail_deletes = True  # 첫 주기들은 Chroma 장애로 실패한다
        with patch.object(rag_service, "PURGE_INTERVAL_SECONDS", 0.01):
            task = asyncio.create_task(rag_service.purge_periodically())
            self.addCleanup(task.cancel)
            await asyncio.sleep(0.05)
            orphan = self._late_write(987_654_321, "ghost")  # 기동 뒤, 삭제 요청 없이 도착한 청크
            await asyncio.sleep(0.05)
            self.assertIn(orphan, self.collection.rows)
            self.assertFalse(task.done())                 # 실패한 주기가 루프를 끝내지 않는다
            self.collection.fail_deletes = False
            await asyncio.sleep(0.2)
        self.assertNotIn(orphan, self.collection.rows)

    async def test_every_delete_runs_the_reconciliation(self):
        analysis_id, doc_id = await self._analyzed_document()
        orphan = self._late_write(987_654_321, "ghost")  # 어떤 문서에도 속하지 않는 청크
        await self._delete_record(analysis_id)
        await rag_service.purge_pending_documents()
        self.assertNotIn(orphan, self.collection.rows)


# ── API ──────────────────────────────────────────────────────────────────


class _ApiCase(_DbCase):
    """실제 앱에 요청을 보내는 테스트의 공통 준비 — 인증·DB 의존성 교체와 가짜 LLM."""

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
                return QaDraft(answerable=True, claims=[Claim(
                    text="GLUE로 평가했다.", citations=[Citation(passage=1, quote="We evaluate on GLUE with RoBERTa")])])

        self.stack.enter_context(patch.object(qa, "_llm", _FakeLLM()))

    async def _ask(self, analysis_id: int, question: str = "glue roberta accuracy benchmark") -> httpx.Response:
        return await self.client.post(f"/api/v1/analyses/{analysis_id}/ask", json={"question": question})


class AskApiTest(_ApiCase):
    async def test_answer_comes_with_verified_page_citations(self):
        analysis_id, _ = await self._analyzed_document()
        response = await self._ask(analysis_id)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["answerable"])
        self.assertEqual(body["citations"], [{"page": 3, "chunk_index": 1, "quote": "We evaluate on GLUE with RoBERTa"}])

    async def test_a_blank_question_is_rejected_before_any_external_call(self):
        analysis_id, _ = await self._analyzed_document()
        self.assertEqual((await self._ask(analysis_id, question="     ")).status_code, 422)
        self.assertEqual(self.embed_calls, [])

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


# ── 분석 결과의 관련 원문 ──────────────────────────────────────────────────


class RelatedItemsTest(unittest.TestCase):
    def test_summary_is_split_only_at_sentence_ends_followed_by_whitespace(self):
        summary = ("LoRA는 드롭아웃 0.1을 쓰고 GPT-3 175B에서 파라미터를 10,000배 줄인다. "
                   "Top-1 정확도는 2.5% 올랐다!\n결과는 W_0 + BA로 쓴다.")
        items = rag_service.related_items(summary, [])
        self.assertEqual([item["id"] for item in items], ["summary-0", "summary-1", "summary-2"])
        self.assertEqual(items[0]["label"], "LoRA는 드롭아웃 0.1을 쓰고 GPT-3 175B에서 파라미터를 10,000배 줄인다.")
        self.assertEqual(items[2]["query"], "결과는 W_0 + BA로 쓴다.")

    def test_formulas_are_searched_by_name_and_description_not_latex(self):
        formulas = [
            {"name": "LoRA Update", "latex": "W = W_0 + BA", "description": "저랭크 행렬의 곱을 더한다."},
            "not a formula",                                  # 저장된 값이 깨져 있어도 건너뛴다
            {"name": "", "latex": "x^2", "description": ""},  # 찾을 말이 없는 수식
            {"name": "Scaling", "latex": "\\alpha / r"},
        ]
        items = rag_service.related_items("", formulas)
        # id는 저장된 목록에서의 순번 — 건너뛴 항목이 있어도 화면의 수식 순번과 맞는다
        self.assertEqual([(item["id"], item["label"]) for item in items], [("formula-0", "LoRA Update"), ("formula-3", "Scaling")])
        self.assertEqual(items[0]["query"], "LoRA Update 저랭크 행렬의 곱을 더한다.")
        self.assertNotIn("W_0", items[0]["query"])

    def test_the_number_and_length_of_queries_are_bounded(self):
        summary = " ".join(f"문장 {i}번이다." for i in range(30))
        formulas = [{"name": f"F{i}", "description": "설" * 2000} for i in range(30)]
        items = rag_service.related_items(summary, formulas)
        self.assertEqual(sum(item["kind"] == "summary" for item in items), rag_service.RELATED_MAX_SUMMARY_SENTENCES)
        self.assertEqual(sum(item["kind"] == "formula" for item in items), rag_service.RELATED_MAX_FORMULAS)
        self.assertTrue(all(len(item["query"]) <= rag_service.RELATED_MAX_QUERY_CHARS for item in items))
        self.assertEqual(rag_service.related_items("", []), [])
        self.assertEqual(rag_service.related_items(None, None), [])


class RelatedApiTest(_ApiCase):
    SUMMARY = "Low rank adaptation freezes the weights. It is evaluated on the GLUE benchmark with RoBERTa."
    FORMULAS = [{"name": "rank decomposition matrices", "latex": "W_0 + BA", "description": "adaptation of weights"}]

    async def _analysis_with_results(self, user_id: int | None = None, pages: list[str] | None = None) -> int:
        analysis_id, _ = await self._analyzed_document(user_id, pages)
        async with self.sessions() as db:
            await db.execute(
                update(AnalysisResult).where(AnalysisResult.id == analysis_id)
                .values(paper_summary=self.SUMMARY, key_formulas=self.FORMULAS)
            )
            await db.commit()
        return analysis_id

    async def _related(self, analysis_id: int) -> httpx.Response:
        return await self.client.get(f"/api/v1/analyses/{analysis_id}/related")

    async def test_each_summary_sentence_and_formula_gets_passages_from_this_document_only(self):
        other_user = await self._new_user()
        self.user_ids.append(other_user)
        _, other_doc = await self._analyzed_document(other_user, ["Low rank adaptation of bananas: GLUE benchmark for bananas."])
        await rag_service.ensure_indexed(await self._document(other_doc))
        analysis_id = await self._analysis_with_results()

        response = await self._related(analysis_id)
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual([(item["id"], item["kind"]) for item in items],
                         [("summary-0", "summary"), ("summary-1", "summary"), ("formula-0", "formula")])
        self.assertEqual(items[2]["label"], "rank decomposition matrices")
        for item in items:
            self.assertTrue(1 <= len(item["passages"]) <= rag_service.RELATED_PASSAGES_PER_ITEM)
            self.assertTrue(all("banana" not in passage["text"] for passage in item["passages"]))
        # 가장 가까운 구절이 먼저 온다: 첫 문장은 1쪽(LoRA), 둘째 문장은 3쪽(GLUE)
        self.assertEqual([items[0]["passages"][0]["page"], items[1]["passages"][0]["page"]], [1, 3])
        # 질의는 저장된 분석에서만 만든다 — 한 번의 임베딩 호출에 항목 3개 (그 앞의 호출들은 색인)
        self.assertEqual(self.embed_calls[-1], [self.SUMMARY.split(". ")[0] + ".", self.SUMMARY.split(". ")[1],
                                                "rank decomposition matrices adaptation of weights"])

    async def test_a_query_sent_by_the_client_is_ignored(self):
        analysis_id = await self._analysis_with_results()
        response = await self.client.get(f"/api/v1/analyses/{analysis_id}/related", params={"query": "bananas", "q": "bananas"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all("bananas" not in text for call in self.embed_calls for text in call))

    async def test_an_analysis_without_summary_or_formulas_returns_no_items(self):
        analysis_id, _ = await self._analyzed_document()
        response = await self._related(analysis_id)
        self.assertEqual((response.status_code, response.json()), (200, {"items": []}))

    async def test_access_and_state_responses_match_the_ask_endpoint(self):
        analysis_id = await self._analysis_with_results()
        other_user = await self._new_user()
        self.user_ids.append(other_user)
        self.current_user = other_user
        self.assertEqual((await self._related(analysis_id)).status_code, 404)

        self.current_user = self.user_id
        async with self.sessions() as db:
            record = await crud_analysis.create_analysis_result(
                db, mode="pdf", query="old.pdf", generated_code="", review_feedback="", review_passed=True,
                iteration_count=1, user_id=self.user_id, document_id=None, paper_summary=self.SUMMARY,
            )
            await db.commit()
        without_document = await self._related(record.id)
        self.assertEqual((without_document.status_code, without_document.json()["reason"]), (409, "no_document"))

        _, doc_id = await self._analyzed_document(pages=["A different paper about something else entirely."])
        indexing_id = (await self._analysis_ids_of(doc_id))[0]
        async with self.sessions() as db:
            await claim_index_job(db, doc_id, stale_after_seconds=90)  # 다른 요청이 색인 중
            await db.commit()
        indexing = await self._related(indexing_id)
        self.assertEqual((indexing.status_code, indexing.json()["retry_after_seconds"]), (202, rag_service.RETRY_AFTER_SECONDS))

        self.app.dependency_overrides.pop(get_current_user)  # 실제 인증 — 토큰 없음
        self.assertEqual((await self._related(analysis_id)).status_code, 401)

    async def _analysis_ids_of(self, document_id: int) -> list[int]:
        async with self.sessions() as db:
            rows = await db.execute(select(AnalysisResult.id).where(AnalysisResult.document_id == document_id))
            return list(rows.scalars())


# ── 실제 HTTP 연결 종료 ───────────────────────────────────────────────────


class ClientDisconnectTest(_DbCase):
    """실제 uvicorn 서버에서 클라이언트가 색인 도중 연결을 끊는다.

    관찰된 동작: 연결이 끊겨도 핸들러는 취소되지 않는다 (ASGITransport나 task.cancel()로는 알 수 없는 부분).
    그래서 색인은 끝까지 돌아 다음 요청이 바로 쓰게 되거나, 멈춰 있다면 시간 제한이 작업을 놓아준다.
    어느 쪽이든 'indexing'으로 굳어 다른 요청을 계속 202로 돌려보내는 일은 없어야 한다.
    """

    async def asyncSetUp(self):
        await super().asyncSetUp()
        from main import app

        async def override_db():
            async with self.sessions() as session:
                yield session
                await session.commit()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=self.user_id)
        self.addCleanup(app.dependency_overrides.clear)

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off"))
        self.serving = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)

    async def asyncTearDown(self):
        self.server.should_exit = True
        await self.serving
        await super().asyncTearDown()

    async def _ask_then_hang_up(self, analysis_id: int, embedding_started: asyncio.Event) -> None:
        body = b'{"question": "what is this paper about?"}'
        _, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(
            b"POST /api/v1/analyses/%d/ask HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (analysis_id, len(body), body)
        )
        await writer.drain()
        await embedding_started.wait()
        writer.close()  # 서버가 임베딩을 기다리는 동안 연결을 끊는다
        await writer.wait_closed()

    async def _wait_until_not_indexing(self, document_id: int, timeout: float) -> PaperDocument:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            document = await self._document(document_id)
            if document.index_status != "indexing" or asyncio.get_running_loop().time() > deadline:
                return document
            await asyncio.sleep(0.05)

    async def test_indexing_finishes_after_the_client_hangs_up_so_the_retry_is_instant(self):
        analysis_id, doc_id = await self._analyzed_document()
        started = asyncio.Event()

        async def slow_embed(texts):
            started.set()
            await asyncio.sleep(0.5)
            return await _fake_embed(texts)

        with patch.object(rag_service, "embed", slow_embed):
            await self._ask_then_hang_up(analysis_id, started)
            document = await self._wait_until_not_indexing(doc_id, timeout=5)

        self.assertEqual(document.index_status, "ready")  # 취소되지 않고 끝까지 색인했다
        self.assertEqual(await rag_service._count_chunks(doc_id, document.index_job_id), 2)

    async def test_a_hung_job_is_released_by_the_time_limit_not_by_the_disconnect(self):
        analysis_id, doc_id = await self._analyzed_document()
        started = asyncio.Event()

        async def hanging_embed(texts):
            started.set()
            await asyncio.sleep(60)

        with patch.object(rag_service, "embed", hanging_embed), patch.object(rag_service, "INDEX_TIME_LIMIT_SECONDS", 1.5):
            await self._ask_then_hang_up(analysis_id, started)
            await asyncio.sleep(0.5)
            self.assertEqual((await self._document(doc_id)).index_status, "indexing")  # 연결 종료만으로는 풀리지 않는다
            document = await self._wait_until_not_indexing(doc_id, timeout=5)

        self.assertEqual(document.index_status, "failed")
        self.assertIn("TimeoutError", document.index_error)
        self.assertEqual(self.collection.rows, {})


# ── 실제 Chroma 서버 ─────────────────────────────────────────────────────


class LifespanPurgeTest(_DbCase):
    """실제 uvicorn + 실제 앱 lifespan으로 주기 정리의 시작·반복·실패 후 재시도·종료 시 취소를 확인한다."""

    SLOW_SECONDS = 0.5

    async def asyncSetUp(self):
        await super().asyncSetUp()
        from main import app

        self.rounds = 0
        self.fail_rounds = 1
        real_list = rag_service.crud_paper_document.list_purge_pending

        async def counting_list(db):
            self.rounds += 1
            if self.rounds <= self.fail_rounds:
                raise ConnectionError("database unavailable")
            return await real_list(db)

        def slow_collection():
            # Chroma 클라이언트 생성은 동기 네트워크 호출이다 — 이벤트 루프에서 불리면 그동안 모든 요청이 멈춘다
            time.sleep(self.SLOW_SECONDS)
            return self.collection

        self.stack.enter_context(patch.object(rag_service.crud_paper_document, "list_purge_pending", counting_list))
        self.stack.enter_context(patch.object(rag_service, "_collection", slow_collection))
        self.stack.enter_context(patch.object(rag_service, "PURGE_INTERVAL_SECONDS", 0.05))

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="on"))
        self.serving = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)

    async def asyncTearDown(self):
        self.server.should_exit = True
        await self.serving
        await super().asyncTearDown()

    async def _wait_for(self, condition, timeout: float = 10) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not condition():
            self.assertLess(asyncio.get_running_loop().time(), deadline, "timed out")
            await asyncio.sleep(0.02)

    async def test_cleanup_starts_repeats_retries_after_a_failure_and_stops_on_shutdown(self):
        orphan = "987654321:ghost:0"
        self.collection.upsert(ids=[orphan], embeddings=[_fake_vector("rank")], documents=["late write"],
                               metadatas=[{"document_id": 987_654_321, "job_id": "ghost", "page": 1, "chunk_index": 0}])

        # 시작: 기동 직후 첫 주기가 돈다. 그 주기는 실패하지만(DB 오류) 루프는 끝나지 않고 다음 주기가 고아 청크를 지운다
        await self._wait_for(lambda: orphan not in self.collection.rows)
        self.assertGreater(self.rounds, self.fail_rounds)

        # 정리가 느린 Chroma 호출에 묶여 있는 동안에도 /health는 바로 응답한다
        # (요청은 다른 스레드에서 보낸다 — 같은 이벤트 루프에서 재면 루프가 멈춘 시간이 측정에서 빠진다)
        def slowest_health_response() -> float:
            slowest = 0.0
            with httpx.Client(base_url=f"http://127.0.0.1:{self.port}") as client:
                for _ in range(15):
                    started = time.perf_counter()
                    self.assertEqual(client.get("/health").status_code, 200)
                    slowest = max(slowest, time.perf_counter() - started)
                    time.sleep(0.07)
            return slowest

        self.assertLess(await asyncio.to_thread(slowest_health_response), self.SLOW_SECONDS / 2)

        # 반복: 주기가 계속 돈다
        seen = self.rounds
        await self._wait_for(lambda: self.rounds >= seen + 2)

        # 종료: 서버가 내려가면 정리 작업이 취소된다 — 더 이상 주기가 돌지 않는다
        self.server.should_exit = True
        started = asyncio.get_running_loop().time()
        await self.serving
        self.assertLess(asyncio.get_running_loop().time() - started, 5)
        stopped_at = self.rounds
        await asyncio.sleep(self.SLOW_SECONDS * 3)
        self.assertEqual(self.rounds, stopped_at)


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

        # 툼스톤이 사라진 뒤에 도착한 쓰기 — 실제 서버에서도 페이지 조회와 id 삭제로 찾아 지운다
        await asyncio.to_thread(
            self.collection.upsert, ids=[f"{doc_id}:{job}:0"], embeddings=[_fake_vector("rank")], documents=["late write"],
            metadatas=[{"document_id": doc_id, "job_id": job, "page": 1, "chunk_index": 0}],
        )
        self.assertEqual(await rag_service.reconcile_orphan_chunks(), 1)
        self.assertEqual(await rag_service._count_chunks(doc_id, job), 0)
        self.assertEqual(await rag_service._count_chunks(other_id, other_job), 1)


if __name__ == "__main__":
    unittest.main()
