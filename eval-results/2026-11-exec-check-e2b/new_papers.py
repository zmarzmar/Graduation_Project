"""운영 분석 그래프(Analyzer → Coder → Executor(E2B) → Reviewer)를 처음 보는 논문 3편에 그대로 돌린다.

실행 (BE에서): caffeinate -i uv run python -u ../eval-results/2026-11-exec-check-e2b/new_papers.py
본 평가(2026-11-01)의 예비 논문 3편 — 이 실행 규칙(실행 검사 도입 후)을 고정한 뒤 처음 돌린다. 결과: new_papers.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from agents.graph import analyze_graph  # noqa: E402
from services.agent_service import _code_result, _make_initial_state, download_pdf_pages  # noqa: E402

PAPERS = {
    "laara": ("2607.19391", "LAARA"),
    "delta_attention_residuals": ("2605.18855", "Delta Attention Residuals"),
    "autonorm": ("2607.10593", "AutoNorm"),
}
OUT = Path("../eval-results/2026-11-exec-check-e2b/new_papers.json")
KEEP = ("iteration_count", "generated_code", "code_skip_reason", "smoke_test", "exec_status", "exec_detail",
        "exec_feedback", "exec_disabled", "review_passed", "review_feedback", "error", "elapsed")


async def run_paper(name: str, arxiv_id: str, title: str) -> dict:
    out: dict = {"paper": name, "arxiv_id": arxiv_id, "events": [], "run_error": None}
    started = time.perf_counter()
    try:
        pages = await download_pdf_pages(f"https://arxiv.org/pdf/{arxiv_id}")
        state = _make_initial_state("analyze", title, pdf_pages=pages)
        state["papers"] = [{"title": title, "arxiv_id": arxiv_id}]
        state["analysis_source"] = "pdf"
        acc = dict(state)
        last = time.perf_counter()
        async with asyncio.timeout(30 * 60):
            async for chunk in analyze_graph.astream(state, stream_mode="updates"):
                for node, upd in chunk.items():
                    now = time.perf_counter()
                    acc.update(upd)
                    ev = {"node": node, "seconds": round(now - last, 1), **{k: upd[k] for k in KEEP if k in upd}}
                    if node == "analyzer":
                        ev["summary_chars"] = len(upd.get("paper_summary", ""))
                    out["events"].append(ev)
                    last = now
                    print(f"  {name} {node:9} {ev['seconds']:6}s {upd.get('exec_status', '')} "
                          f"{'' if 'review_passed' not in upd else ('검토통과' if upd['review_passed'] else '낙제')} {upd.get('error') or ''}",
                          flush=True)
        out["final"] = _code_result(acc)
    except Exception as e:
        out["run_error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


async def main() -> None:
    results = []
    for name, (arxiv_id, title) in PAPERS.items():
        print(f"{name} 실행 중...", flush=True)
        results.append(await run_paper(name, arxiv_id, title))
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for r in results:
        f = r.get("final") or {}
        print(r["paper"], r["seconds"], "s |", r["run_error"] or "", "| 회차", f.get("iteration_count"),
              "| 실행", f.get("exec_status"), "| 검토", f.get("review_passed"), "| 오류", f.get("error"))


asyncio.run(main())
