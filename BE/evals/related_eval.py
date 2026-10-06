"""분석 결과의 '관련 원문' 검색 평가 — 실제 Analyzer·임베딩·Chroma로 돌린다 (API 비용 발생, CI에서는 실행하지 않는다).

실행: cd BE && uv run python -m evals.related_eval [--out results.json]

자동 채점은 하지 않는다. 항목(요약 문장·수식)과 검색된 구절을 그대로 출력하니 사람이 읽고
'구절이 그 항목과 관련 있는가'를 직접 판정한다 — 페이지 번호가 맞아도 구절은 엉뚱할 수 있다.
분석 결과는 한국어, 논문은 영어다: 언어가 다른 검색의 품질을 보는 것이 이 평가의 목적이다.
"""

import argparse
import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from agents.nodes.analyzer import analyzer_node
from core.config import settings
from services import rag_service
from services.agent_service import download_pdf_pages, join_pages

PAPERS = {
    "lora": "https://arxiv.org/pdf/2106.09685",
    "attention": "https://arxiv.org/pdf/1706.03762",
}


async def run(out: str | None) -> None:
    collection_name = f"paper_chunks_eval_{uuid.uuid4().hex[:10]}"
    rag_service._collection.cache_clear()
    rows = []
    with patch.object(rag_service, "_COLLECTION", collection_name):
        try:
            for number, (name, url) in enumerate(PAPERS.items()):
                pages = await download_pdf_pages(url)
                analysis = await analyzer_node({"pdf_text": join_pages(pages), "papers": []})
                # DB 없이 색인한다 — 문서 id는 이 실행에서만 쓰는 임의의 값
                document = SimpleNamespace(id=910_000 + number, pages=pages, indexed_chunk_count=0)
                job_id = uuid.uuid4().hex
                document.indexed_chunk_count = await rag_service._run_index_job(document, job_id)
                summary, formulas = analysis["paper_summary"], analysis["key_formulas"]
                items = await rag_service.find_related(document, job_id, summary, formulas)
                # 실제로 검색에 쓴 질의를 함께 남긴다 (수식은 이름 + 설명이라 label만으로는 알 수 없다)
                queries = {item["id"]: item["query"] for item in rag_service.related_items(summary, formulas)}
                rows.extend({"paper": name, "query": queries[item["id"]], **item} for item in items)
        finally:
            client = rag_service.chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            await asyncio.to_thread(client.delete_collection, collection_name)
            rag_service._collection.cache_clear()

    for row in rows:
        print(f"\n[{row['paper']}] {row['id']} — 질의: {row['query']}")
        for rank, passage in enumerate(row["passages"], start=1):
            text = " ".join(passage["text"].split())
            print(f"  {rank}. p.{passage['page']} ({len(text)}자): {text[:400]}")
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="결과를 JSON으로 저장할 경로")
    asyncio.run(run(parser.parse_args().out))
