"""자기수정 루프 평가 — 실제 LLM으로 분석 그래프를 돌린다 (API 비용 발생, CI에서는 실행하지 않는다).

실행: cd BE && uv run python -m evals.loop_eval --out results.json [--papers attention lora]

논문마다 분석 그래프(Analyzer → Coder ↔ Reviewer)를 한 번 돌리고, 회차마다 그 회차의 코드·정적 검사·
Reviewer 판정과 피드백을 남긴다. 그래프 상태는 회차마다 덮어써지므로 노드가 끝날 때 나오는 갱신분을 모은다.

이 스크립트가 말해 주지 않는 것
- 정적 검사는 문법과 import 목록까지다. 코드를 실행하지 않는다 (evals/static_check.py)
- Reviewer의 통과는 'LLM 검토 통과'이지 검증 통과가 아니다
- 논문당 1회 실행이다. 같은 논문도 실행마다 결과가 달라질 수 있다
- 최초본과 수정본이 논문의 요구를 채우는지는 여기서 판정하지 않는다 — 결과의 코드를 체크리스트와 대조해 사람이 본다
"""

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Iterable

from agents.graph import analyze_graph
from agents.nodes import analyzer, coder, reviewer
from evals.static_check import static_check
from services.agent_service import _make_initial_state, download_pdf_pages

PAPERS: dict[str, dict] = {
    "attention": {"title": "Attention Is All You Need", "pdf_url": "https://arxiv.org/pdf/1706.03762"},
    "lora": {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "pdf_url": "https://arxiv.org/pdf/2106.09685"},
}

# agents/nodes/router.py의 반복 상한과 같아야 한다 (테스트가 route()와 맞는지 확인한다)
MAX_ITERATIONS = 3


def collect_rounds(events: Iterable[tuple[str, dict]]) -> list[dict]:
    """(노드 이름, 갱신분) 순서열을 회차 목록으로 묶는다. Coder의 갱신이 회차를 열고 Reviewer의 갱신이 닫는다."""
    rounds: list[dict] = []
    for node, updates in events:
        if node == "coder":
            code = updates.get("generated_code", "")
            rounds.append({
                "iteration": updates.get("iteration_count"),
                "code": code,
                "coder_error": updates.get("error"),
                "static": static_check(code),
                "reviewed": False,
                "review_passed": None,
                "review_feedback": "",
                "reviewer_error": None,
            })
        elif node == "reviewer":
            rounds[-1].update(
                reviewed=True,
                review_passed=updates.get("review_passed", False),
                review_feedback=updates.get("review_feedback", ""),
                reviewer_error=updates.get("error"),
            )
    return rounds


def end_reason(rounds: list[dict], run_error: str | None) -> str:
    """루프가 끝난 이유. 회차 안의 오류(coder_error·reviewer_error)는 회차에 따로 남는다."""
    if run_error:
        return "error"
    if rounds and rounds[-1]["review_passed"]:
        return "passed"
    if len(rounds) >= MAX_ITERATIONS and rounds[-1]["reviewed"]:
        return "max_iterations"
    return "incomplete"


async def run_paper(name: str) -> dict:
    paper = PAPERS[name]
    pages: list[str] = []
    events: list[tuple[str, dict]] = []
    run_error: str | None = None
    started = time.perf_counter()
    # 다운로드나 그래프가 실패해도 그때까지 모은 회차와 실패 사유를 결과로 남긴다
    try:
        pages = await download_pdf_pages(paper["pdf_url"])
        # /agent/analyze가 전문을 받은 경우와 같은 초기 상태
        state = _make_initial_state("analyze", paper["title"], pdf_pages=pages)
        state["papers"] = [paper]
        state["analysis_source"] = "pdf"
        async for chunk in analyze_graph.astream(state, stream_mode="updates"):
            for node, updates in chunk.items():
                events.append((node, updates))
                print(f"  [{name}] {node} 완료 ({time.perf_counter() - started:.0f}s)")
    except Exception as e:
        run_error = f"{type(e).__name__}: {e}"

    rounds = collect_rounds(events)
    analysis = next((updates for node, updates in events if node == "analyzer"), {})
    return {
        "paper": name,
        "title": paper["title"],
        "pages": len(pages),
        "seconds": round(time.perf_counter() - started, 1),
        "end_reason": end_reason(rounds, run_error),
        "run_error": run_error,
        # Analyzer는 실패해도 그래프가 Coder로 계속 간다 — 요약·수식 없이 생성된 것인지 알 수 있게 남긴다
        "analyzer_error": analysis.get("error"),
        "paper_summary": analysis.get("paper_summary", ""),
        "key_formulas": analysis.get("key_formulas", []),
        "rounds": rounds,
    }


def report(results: list[dict]) -> None:
    for result in results:
        print(f"\n{result['paper']}: {result['end_reason']} — {len(result['rounds'])}회차, {result['seconds']}s")
        for key in ("run_error", "analyzer_error"):
            if result[key]:
                print(f"  {key}: {result[key]}")
        for r in result["rounds"]:
            static = r["static"]
            syntax = "문법 통과" if static["syntax_ok"] else f"문법 오류({static['syntax_error']})"
            verdict = {True: "LLM 검토 통과", False: "낙제", None: "검토 안 됨"}[r["review_passed"]]
            errors = "".join(f" / {key}: {r[key]}" for key in ("coder_error", "reviewer_error") if r[key])
            print(f"  {r['iteration']}회차: {len(r['code'])}자, {syntax}, "
                  f"확인 필요한 import {static['unverified_imports']}, {verdict}{errors}")


async def run(names: list[str], out: str | None) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"commit": commit, "models": {"analyzer": analyzer._MODEL, "coder": coder._MODEL, "reviewer": reviewer._MODEL}}
    results = []
    # 한 편씩 돌린다 — 루프 3회면 Coder·Reviewer 모델의 분당 토큰 한도에 가깝다 (o4-mini 때 측정)
    for name in names:
        print(f"{name} 실행 중...")
        results.append(await run_paper(name))
        if out:
            # 논문이 끝날 때마다 저장한다 — 뒤의 논문에서 멈춰도 비용을 들인 결과가 남는다
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    report(results)
    if out:
        print(f"\n저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", choices=list(PAPERS), default=list(PAPERS))
    parser.add_argument("--out", help="결과(회차별 코드 포함)를 JSON으로 저장할 경로")
    args = parser.parse_args()
    asyncio.run(run(args.papers, args.out))
