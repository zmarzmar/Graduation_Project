"""요약 문장별 원문 인용 — 인용 확인 통과율을 모델별로 잰다 (실제 API 비용 발생, CI에서 돌리지 않는다).

실제 analyzer_node를 그대로 부르고 모델만 바꾼다. 재는 것은 '인용이 그 페이지에 실제로 있는가'뿐이다 —
인용이 문장을 뒷받침하는지는 재지 않는다.

    caffeinate -i uv run python -u -m evals.summary_cite_eval --out <파일>
"""

import argparse
import asyncio
import json
import time

from langchain_openai import ChatOpenAI

import agents.nodes.analyzer as analyzer
from agents.token_budget import ANALYZER_OUTPUT_BUDGET
from core.config import settings
from evals.loop_eval import PAPERS
from services.agent_service import _make_initial_state, download_pdf_pages

DEFAULT_PAPERS = ["attention", "lora", "avq_attention"]
MODELS = {
    "gpt-4o-mini": {},
    "gpt-5.4-mini-low": {"model": "gpt-5.4-mini", "reasoning_effort": "low"},
}


def _llm(name: str) -> ChatOpenAI:
    extra = MODELS[name]
    return ChatOpenAI(
        model=extra.get("model", name),
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_analyzer_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=ANALYZER_OUTPUT_BUDGET,
        model_kwargs={"response_format": {"type": "json_object"}},
        **({"reasoning_effort": extra["reasoning_effort"]} if "reasoning_effort" in extra else {}),
    )


async def run_once(model: str, title: str, pages: list[str]) -> dict:
    analyzer._llm = _llm(model)
    analyzer._MODEL = MODELS[model].get("model", model)  # ensure_request_fits가 이 이름으로 한도를 찾는다
    state = _make_initial_state("analyze", title, pdf_pages=pages)
    started = time.perf_counter()
    out = await analyzer.analyzer_node(state)
    claims = out.get("summary_claims", [])
    return {
        "seconds": round(time.perf_counter() - started, 1),
        "error": out.get("error"),
        "claims": claims,
        "verified": sum(claim["verified"] for claim in claims),
        "total": len(claims),
        # 모델을 바꾸면 리뷰·수식도 바뀐다 — 함께 남겨 사람이 읽고 비교한다
        "paper_review": out.get("paper_review", {}),
        "key_formulas": out.get("key_formulas", []),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--papers", nargs="+", choices=list(PAPERS), default=DEFAULT_PAPERS)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    results: list[dict] = []
    for name in args.papers:
        pages = await download_pdf_pages(PAPERS[name]["pdf_url"])
        for model in args.models:
            for run in range(1, args.runs + 1):
                try:
                    result = await run_once(model, PAPERS[name]["title"], pages)
                except Exception as e:  # 한 번의 실패가 나머지 측정을 버리지 않게
                    result = {"error": f"{type(e).__name__}: {e}", "claims": [], "verified": 0, "total": 0}
                results.append({"paper": name, "model": model, "run": run, **result})
                print(f"{name} {model} #{run}: {result['verified']}/{result['total']} 확인 "
                      f"({result.get('seconds', '-')}s) {result['error'] or ''}")
                with open(args.out, "w") as f:  # 한 번 끝날 때마다 저장
                    json.dump(results, f, ensure_ascii=False, indent=2)

    for model in args.models:
        rows = [r for r in results if r["model"] == model]
        verified, total = sum(r["verified"] for r in rows), sum(r["total"] for r in rows)
        print(f"[합계] {model}: 문장 {verified}/{total} 확인, 오류 {sum(bool(r['error']) for r in rows)}회")


if __name__ == "__main__":
    asyncio.run(main())
