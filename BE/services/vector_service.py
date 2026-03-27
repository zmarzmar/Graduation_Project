import logging

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from core.config import settings

logger = logging.getLogger(__name__)

# ChromaDB 클라이언트 (싱글톤)
_chroma_client = None  # chromadb.HttpClient 인스턴스
_collection = None

# OpenAI 임베딩 클라이언트
_openai_client = OpenAI(api_key=settings.openai_api_key)

_COLLECTION_NAME = "papers"
_EMBED_MODEL = "text-embedding-3-small"


def _get_client() -> chromadb.HttpClient:
    """ChromaDB HTTP 클라이언트를 반환한다 (지연 초기화)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_collection():
    """papers 컬렉션을 반환한다 (없으면 생성)."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환한다."""
    response = _openai_client.embeddings.create(
        model=_EMBED_MODEL,
        input=text[:8000],  # 토큰 제한
    )
    return response.data[0].embedding


def upsert_paper(arxiv_id: str, title: str, abstract: str) -> None:
    """논문 임베딩을 ChromaDB에 저장한다 (이미 있으면 업데이트)."""
    try:
        collection = _get_collection()
        text = f"{title}\n\n{abstract}"
        embedding = _embed(text)
        collection.upsert(
            ids=[arxiv_id],
            embeddings=[embedding],
            metadatas=[{"title": title, "arxiv_id": arxiv_id}],
            documents=[text],
        )
        logger.info(f"[VectorService] 논문 임베딩 저장 완료: {arxiv_id}")
    except Exception as e:
        logger.error(f"[VectorService] 임베딩 저장 실패: {e}")


def search_similar(query: str, n_results: int = 5) -> list[dict]:
    """쿼리와 유사한 논문을 검색한다."""
    try:
        collection = _get_collection()
        query_embedding = _embed(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        papers = []
        for meta, dist in zip(
            results["metadatas"][0], results["distances"][0]
        ):
            papers.append({**meta, "score": round(1 - dist, 4)})
        return papers
    except Exception as e:
        logger.error(f"[VectorService] 유사 논문 검색 실패: {e}")
        return []
