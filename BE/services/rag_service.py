"""논문 Q&A의 검색 계층 — 청크 분할, 첫 질문 때 색인, 문서 범위 검색, 삭제된 문서의 색인 정리.

Postgres(paper_documents)가 원본이고 Chroma는 거기서 다시 만들 수 있는 파생 색인이다.
DB 트랜잭션이나 사용자 문서 락을 쥔 채로 임베딩·Chroma를 호출하지 않는다 — 상태 변경은 짧게 커밋하고 나서 호출한다.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import chromadb
from openai import AsyncOpenAI

from agents.qa import answer_question, no_evidence
from agents.token_budget import _encoding, count_tokens
from core.config import settings
from core.dependencies import AsyncSessionLocal
from crud import paper_document as crud_paper_document
from models.paper_document import PaperDocument

logger = logging.getLogger(__name__)

_COLLECTION = "paper_chunks"
CHUNK_TOKENS = 500            # 청크 목표 크기
_HARD_SPLIT_OVERLAP = 50      # 문장 하나가 청크보다 길어 토큰 단위로 자를 때의 겹침
_RETOKENIZE_MARGIN = 8
_EMBED_BATCH = 96
_RECONCILE_PAGE = 1000

INDEX_TIME_LIMIT_SECONDS = 60   # 첫 질문 요청 안에서 색인에 쓸 수 있는 시간
INDEX_STALE_AFTER_SECONDS = 90  # 이보다 오래된 'indexing'은 죽은 작업으로 보고 다시 선점 (시간 제한보다 길게)
RETRY_AFTER_SECONDS = 3         # 색인 중일 때 클라이언트에 알려주는 재시도 간격
# 툼스톤 행은 이 시간이 지난 뒤에야 지운다. 색인 중에 삭제된 문서는 늦게 도착한 upsert가 청크를 다시 만들 수 있어서,
# 색인 시간 제한보다 길게 기다린 뒤 한 번 더 지우고 나서 행을 없앤다.
PURGE_GRACE_SECONDS = 120


class IndexingInProgress(Exception):
    """다른 요청이 이 문서를 색인하는 중이다. 잠시 뒤 다시 시도하면 된다."""


@dataclass(frozen=True)
class Chunk:
    page: int    # 1부터 시작하는 원본 PDF 페이지 번호
    index: int   # 문서 안에서의 순번
    text: str


@dataclass(frozen=True)
class Passage:
    chunk_index: int
    page: int
    text: str
    distance: float


# ── 청크 분할 ─────────────────────────────────────────────────────────────


def _split_long_text(text: str, limit: int) -> list[str]:
    """limit 토큰을 넘는 문단을 나눈다: 먼저 문장 경계에서, 문장 하나가 limit보다 길면 토큰 단위로 겹쳐서 자른다."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?。])\s+", text):
        if not sentence.strip():
            continue
        if count_tokens(sentence) <= limit:
            pieces.append(sentence)
            continue
        # 문장 경계가 없는 긴 덩어리(표·수식 나열 등)
        tokens = _encoding().encode(sentence, disallowed_special=())
        # 잘라서 디코딩한 조각은 경계에서 다시 토큰화되며 몇 토큰 늘 수 있다 — 여유를 두고 자른다
        window = limit - _RETOKENIZE_MARGIN
        step = window - _HARD_SPLIT_OVERLAP
        pieces.extend(_encoding().decode(tokens[start:start + window]) for start in range(0, len(tokens), step))
    return pieces


def chunk_pages(pages: list[str], limit: int = CHUNK_TOKENS) -> list[Chunk]:
    """페이지별 본문을 limit 토큰 이하의 청크로 나눈다. 같은 입력은 항상 같은 청크를 만든다.

    청크는 페이지를 넘지 않는다 → 청크 하나가 페이지 번호 하나에 대응해 출처 표시가 정확하다.
    ponytail: 페이지 경계에 걸친 문단은 둘로 나뉜다. 평가에서 문제가 되면 앞뒤 페이지 문맥을 덧붙이는 방식으로 보완한다.
    """
    chunks: list[Chunk] = []
    for page_number, page in enumerate(pages, start=1):
        pieces: list[str] = []
        for paragraph in re.split(r"\n\s*\n", page):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            pieces.extend([paragraph] if count_tokens(paragraph) <= limit else _split_long_text(paragraph, limit))

        current: list[str] = []
        for piece in pieces:
            if current and count_tokens("\n".join([*current, piece])) > limit:
                chunks.append(Chunk(page_number, len(chunks), "\n".join(current)))
                current = []
            current.append(piece)
        if current:
            chunks.append(Chunk(page_number, len(chunks), "\n".join(current)))
    return chunks


# ── 외부 호출 (임베딩·Chroma) ──────────────────────────────────────────────


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    # 임베딩은 서버가 계산해서 넘긴다 — Chroma의 기본 임베딩 함수(로컬 ONNX 모델)는 쓰지 않는다
    return client.get_or_create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


@lru_cache(maxsize=1)
def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        response = await _openai().embeddings.create(model=settings.embedding_model, input=texts[start:start + _EMBED_BATCH])
        vectors.extend(item.embedding for item in response.data)
        logger.info(f"[RAG] 임베딩 {len(response.data)}건, {response.usage.total_tokens} 토큰")
    return vectors


def _job_filter(document_id: int, job_id: str) -> dict:
    return {"$and": [{"document_id": document_id}, {"job_id": job_id}]}


async def _count_chunks(document_id: int, job_id: str) -> int:
    found = await asyncio.to_thread(_collection().get, where=_job_filter(document_id, job_id), include=[])
    return len(found["ids"])


async def _delete_chunks(where: dict) -> None:
    await asyncio.to_thread(_collection().delete, where=where)


# ── 색인 ─────────────────────────────────────────────────────────────────


async def _run_index_job(document: PaperDocument, job_id: str) -> int:
    chunks = chunk_pages(document.pages)
    if not chunks:
        return 0
    vectors = await embed([chunk.text for chunk in chunks])
    await asyncio.to_thread(
        _collection().upsert,
        ids=[f"{document.id}:{job_id}:{chunk.index}" for chunk in chunks],
        embeddings=vectors,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"document_id": document.id, "job_id": job_id, "page": chunk.page, "chunk_index": chunk.index}
            for chunk in chunks
        ],
    )
    # 청크 몇 개가 있다고 완료로 보지 않는다 — 기대한 개수가 모두 저장됐는지 확인한다
    stored = await _count_chunks(document.id, job_id)
    if stored != len(chunks):
        raise RuntimeError(f"색인 검증 실패: {len(chunks)}개 중 {stored}개만 저장됨")
    return len(chunks)


async def _abandon_job(document_id: int, job_id: str, error: str) -> None:
    """실패·시간 초과·취소·대체된 작업의 뒷정리. 현재 작업일 때만 failed로 기록하고, 자기 청크만 지운다."""
    try:
        async with AsyncSessionLocal() as db:
            await crud_paper_document.fail_index_job(db, document_id, job_id, error)
            await db.commit()
        await _delete_chunks(_job_filter(document_id, job_id))
    except Exception as e:
        # 남은 청크는 검색 대상이 아니고(현재 job_id가 아님), 문서 삭제 때 document_id 기준으로 지워진다
        logger.error(f"[RAG] 색인 작업 뒷정리 실패 doc={document_id} job={job_id}: {e}")


async def ensure_indexed(document: PaperDocument) -> str:
    """문서가 검색 가능한 상태인지 보장하고 검색에 쓸 job_id를 반환한다. 다른 요청이 색인 중이면 IndexingInProgress."""
    if document.index_status == "ready" and document.index_job_id:
        return document.index_job_id

    async with AsyncSessionLocal() as db:
        job_id = await crud_paper_document.claim_index_job(db, document.id, INDEX_STALE_AFTER_SECONDS)
        await db.commit()  # 선점만 커밋하고 끝낸다 — 임베딩 동안 트랜잭션을 열어두지 않는다
    if job_id is None:
        raise IndexingInProgress()

    try:
        async with asyncio.timeout(INDEX_TIME_LIMIT_SECONDS):
            chunk_count = await _run_index_job(document, job_id)
        async with AsyncSessionLocal() as db:
            finished = await crud_paper_document.finish_index_job(db, document.id, job_id, chunk_count)
            await db.commit()
    # 태스크 취소(CancelledError — 서버 종료 등)에도 뒷정리를 한다. 클라이언트가 연결을 끊는 것만으로는
    # 핸들러가 취소되지 않는다(실측): 색인은 끝까지 돌아 다음 요청이 쓰거나, 멈췄다면 위의 시간 제한이 놓아준다.
    except BaseException as e:
        await asyncio.shield(_abandon_job(document.id, job_id, f"{type(e).__name__}: {e}"))
        raise

    if not finished:
        # 그 사이 문서가 삭제됐거나 다른 작업이 선점했다 — 이 작업의 청크는 쓰이지 않으므로 지운다
        await _abandon_job(document.id, job_id, "superseded")
        raise IndexingInProgress()

    # 이전 작업들이 남긴 청크 정리 (검색에는 영향 없고 공간만 차지한다)
    try:
        await _delete_chunks({"$and": [{"document_id": document.id}, {"job_id": {"$ne": job_id}}]})
    except Exception as e:
        logger.warning(f"[RAG] 이전 색인 청크 정리 실패 (무시) doc={document.id}: {e}")
    return job_id


# ── 검색 ─────────────────────────────────────────────────────────────────


async def retrieve(document: PaperDocument, job_id: str, question: str) -> list[Passage]:
    """질문과 가까운 청크를 '이 문서의 현재 색인' 안에서만 찾는다. 범위 필터는 서버가 정한다."""
    [vector] = await embed([question])
    found = await asyncio.to_thread(
        _collection().query,
        query_embeddings=[vector],
        n_results=settings.qa_top_k,
        where=_job_filter(document.id, job_id),
        include=["documents", "metadatas", "distances"],
    )
    passages = [
        Passage(chunk_index=meta["chunk_index"], page=meta["page"], text=text, distance=distance)
        for text, meta, distance in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
    ]

    # 색인 유실과 '근거 없음'을 구분한다: 거리 기준을 적용하기 전인데도 아무것도 안 나왔고,
    # 있어야 할 청크가 실제로 없을 때만 다시 색인하게 되돌린다. 근거가 부족한 질문 때문에 재색인하지 않는다.
    if not passages and (document.indexed_chunk_count or 0) > 0 and await _count_chunks(document.id, job_id) == 0:
        async with AsyncSessionLocal() as db:
            await crud_paper_document.reset_lost_index(db, document.id, job_id)
            await db.commit()
        logger.warning(f"[RAG] 색인 유실 감지 → 재색인 예정 doc={document.id}")
        raise IndexingInProgress()
    return passages


# ── 질문에 답하기 ─────────────────────────────────────────────────────────


async def ask(document: PaperDocument, question: str) -> dict:
    """문서 한 편에 대한 질문에 답한다. 색인이 없으면 이 요청 안에서 만든다 (다른 요청이 만드는 중이면 IndexingInProgress)."""
    job_id = await ensure_indexed(document)
    passages = await retrieve(document, job_id, question)
    # 가장 가까운 구절조차 질문과 멀면 LLM을 부르지 않는다 — 비슷해 보이는 구절을 억지로 근거로 붙이지 않는다
    if not passages or passages[0].distance > settings.qa_max_distance:
        return no_evidence()
    return await answer_question(question, passages)


# ── 삭제된 문서의 색인 정리 ────────────────────────────────────────────────


async def purge_pending_documents() -> None:
    """툼스톤이 된 문서의 청크를 지우고, 유예 시간이 지난 툼스톤 행을 제거한다. 몇 번을 다시 실행해도 안전하다.

    Postgres와 Chroma는 한 트랜잭션이 아니다 — Chroma 삭제가 실패하면 툼스톤(원문은 이미 비워짐)이 남아
    다음 실행에서 다시 시도한다. 행은 청크 삭제가 성공한 뒤에만 지운다.
    """
    async with AsyncSessionLocal() as db:
        pending = await crud_paper_document.list_purge_pending(db)
    removable_before = datetime.now(timezone.utc) - timedelta(seconds=PURGE_GRACE_SECONDS)

    for document_id, marked_at in pending:
        try:
            await _delete_chunks({"document_id": document_id})
        except Exception as e:
            logger.error(f"[RAG] 색인 삭제 실패 — 툼스톤을 남기고 다음에 다시 시도 doc={document_id}: {e}")
            continue
        if marked_at <= removable_before:
            async with AsyncSessionLocal() as db:
                await crud_paper_document.delete_purged_document(db, document_id)
                await db.commit()

    # 유예 시간은 늦은 쓰기를 '대부분' 걸러낼 뿐이다 — 요청을 취소해도 Chroma가 이미 받은 쓰기까지 취소된다는
    # 보장은 없다. 툼스톤이 사라진 뒤에 도착한 청크는 아래 대조 정리가 찾아서 지운다.
    try:
        await reconcile_orphan_chunks()
    except Exception as e:
        logger.error(f"[RAG] 고아 청크 정리 실패 (다음에 다시 시도): {e}")


async def reconcile_orphan_chunks() -> int:
    """벡터 색인을 Postgres와 대조해 주인 없는 청크를 지우고, 지운 개수를 반환한다. 시간 기준에 기대지 않는다.

    고아 청크 = 문서 행이 없거나(삭제가 끝난 문서에 늦게 도착한 쓰기) 삭제 대기 중이거나,
    그 문서의 현재 index_job_id가 아닌 작업이 쓴 청크(대체되거나 죽은 작업의 잔여물).

    순서가 중요하다: Chroma를 먼저 훑고 '그 다음에' Postgres를 읽는다. 색인 작업은 선점을 커밋한 뒤에야
    청크를 쓰므로, 훑을 때 보인 청크의 작업은 그 뒤에 읽은 Postgres에 반드시 있다 — 진행 중인 색인의 청크를
    고아로 오판하지 않는다.

    ponytail: 컬렉션 전체를 훑는다(O(전체 청크 수)). 삭제·서버 기동 때만 돌아서 지금 규모에선 충분하다.
    느려지면 문서별 청크 수를 Postgres에 두고 달라진 문서만 확인하는 방식으로 바꾼다.
    """
    seen: list[tuple[str, int | None, str | None]] = []
    offset = 0
    while True:
        page = await asyncio.to_thread(_collection().get, include=["metadatas"], limit=_RECONCILE_PAGE, offset=offset)
        if not page["ids"]:
            break
        seen.extend(
            (chunk_id, meta.get("document_id"), meta.get("job_id"))
            for chunk_id, meta in zip(page["ids"], page["metadatas"])
        )
        offset += len(page["ids"])

    async with AsyncSessionLocal() as db:
        live_jobs = await crud_paper_document.list_live_index_jobs(db)

    orphans = [
        chunk_id for chunk_id, document_id, job_id in seen
        if document_id not in live_jobs or live_jobs[document_id] != job_id
    ]
    for start in range(0, len(orphans), _RECONCILE_PAGE):
        await asyncio.to_thread(_collection().delete, ids=orphans[start:start + _RECONCILE_PAGE])
    if orphans:
        logger.warning(f"[RAG] 고아 청크 {len(orphans)}개 정리")
    return len(orphans)
