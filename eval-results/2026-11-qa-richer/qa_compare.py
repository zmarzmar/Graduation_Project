"""조사용: Attention 논문으로 Q&A 답변 길이·빠지는 문장 수를 프롬프트별로 비교 (운영 코드 변경 없음, 실제 API).
실행: cd BE && PYTHONPATH=. uv run python <이 파일> <label>"""
import asyncio, json, os, sys, time
import numpy as np
from agents import qa
from services.agent_service import download_pdf_pages
from services.rag_service import Passage, chunk_pages, embed

QUESTIONS = ["핵심 연산이 뭐야?", "Multi-head attention은 왜 쓰는 거야?", "위치 정보는 어떻게 넣어?",
             "self-attention이 RNN보다 좋은 점이 뭐야?", "학습은 어떤 설정으로 했어?"]
RUNS = 2

async def main(label: str) -> None:
    if os.environ.get("QA_MODEL"):
        from langchain_openai import ChatOpenAI
        from core.config import settings
        qa._llm = ChatOpenAI(model=os.environ["QA_MODEL"], reasoning_effort=os.environ.get("QA_EFFORT", "low"),
                             api_key=settings.openai_api_key, timeout=120, max_retries=1,
                             max_tokens=qa.QA_OUTPUT_BUDGET * 2).with_structured_output(qa.QaDraft)
    pages = await download_pdf_pages("https://arxiv.org/pdf/1706.03762")
    chunks = chunk_pages(pages)
    vecs = np.array(await embed([c.text for c in chunks]))
    qv = np.array(await embed(QUESTIONS))
    rows = []
    for q, v in zip(QUESTIONS, qv):
        sims = vecs @ v / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(v))
        top = np.argsort(-sims)[:6]
        passages = [Passage(chunk_index=chunks[i].index, page=chunks[i].page, text=chunks[i].text, distance=float(1 - sims[i])) for i in top]
        for r in range(RUNS):
            t0 = time.perf_counter()
            res = await qa.answer_question(q, passages)
            print(f'  {time.perf_counter()-t0:.1f}초', flush=True)
            rows.append({"q": q, "run": r, "answerable": res["answerable"], "claims": len(res["claims"]),
                         "chars": len(res["answer"]) if res["answerable"] else 0, "dropped_claims": res["dropped_claims"],
                         "answer": res["answer"], "claims_full": res["claims"]})
            print(f"[{label}] {q} #{r}: claims {len(res['claims'])}, {rows[-1]['chars']}자, 빠짐 {res['dropped_claims']}", flush=True)
    n = len(rows)
    print(f"[{label}] 합계: 답함 {sum(r['answerable'] for r in rows)}/{n}, 평균 문장 {sum(r['claims'] for r in rows)/n:.1f}, "
          f"평균 {sum(r['chars'] for r in rows)/n:.0f}자, 빠진 문장 합 {sum(r['dropped_claims'] for r in rows)}")
    claims = [{"q": r["q"], "text": c["text"], "quotes": [x["quote"] for x in c["citations"]]} for r in rows for c in r["claims_full"]]
    json.dump(claims, open(f"eval-results/2026-11-qa-richer/qa_{label}_claims.json", "w"), ensure_ascii=False, indent=1)
    json.dump(rows, open(f"eval-results/2026-11-qa-richer/qa_{label}.json", "w"), ensure_ascii=False, indent=1)

asyncio.run(main(sys.argv[1]))
