"""실행 검사 피드백 실험 (평가 전용, 실제 LLM·Docker 사용, CI에서는 실행하지 않는다).

실행: cd BE && caffeinate -i uv run python -u -m evals.exec_loop_eval --out <파일> [--papers ...]
먼저 이미지를 만든다: docker build -t paperpilot-exec-check:dev -f evals/exec_check/Dockerfile .

논문마다 Analyzer → Coder 1회차(smoke_test 포함 작성) → 실행 검사까지는 한 번만 돌리고, 같은 최초 코드에서 두 조건으로 갈라 진행한다.
- exec    — 실행 검사가 실패하면 그 결과를 Coder에게 돌려준다 (Reviewer 건너뜀). 통과하면 Reviewer → 지금과 같은 루프
- control — 지금 루프 그대로 (Reviewer만 피드백). 실행 검사는 회차마다 기록만 한다
두 조건 모두 첫 작성 포함 최대 3회. smoke_test는 처음 나온 회차의 것으로 고정하고, 이후 Coder가 다시 써도 버린다.

Coder 지시문의 실행 검사 규칙은 2026-11 이후 운영에도 들어갔다 — 실험 당시(#60)에는 이 실험에서만 덧붙였다.
이 실험이 말해 주지 않는 것
- 실행 통과는 'Coder가 만든 smoke_test의 해당 경로가 이 환경에서 돌았다'까지다. 논문대로 구현했다는 뜻이 아니다
- smoke_test도 Coder가 쓰므로 쉬운 테스트일 수 있다
- 논문당 1회 실행이다
"""

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Awaitable, Callable

from agents.exec_check import SMOKE_RULES
from agents.nodes import coder, reviewer
from agents.nodes.analyzer import analyzer_node
from evals import exec_check
from evals.loop_eval import MAIN_PAPERS, MAX_ITERATIONS, PAPERS
from evals.static_check import static_check
from services.agent_service import _make_initial_state, download_pdf_pages

ARMS = ("exec", "control")
PAPER_TIMEOUT_SECONDS = 45 * 60  # 두 조건을 이어서 돌린다

# 실행 검사 규칙은 운영 Coder 지시문에 들어갔다 (agents/exec_check.SMOKE_RULES) — 이 실험도 같은 지시문을 쓴다


class InfraError(Exception):
    """실행 환경 문제 — 코드 탓이 아니므로 그 조건의 실험을 멈춘다."""


class LLMError(Exception):
    """Coder·Reviewer 호출 실패 — 판정 실패와 섞지 않고 그 조건의 실험을 멈춘다."""


def _combined(impl: str, test_src: str | None) -> str:
    return impl.rstrip() + ("\n\n\n" + test_src.strip() + "\n" if test_src else "\n")


async def _checked(impl: str, test_src: str | None, run_exec: Callable[..., Awaitable[dict]]) -> dict:
    result = await run_exec(impl, test_src)
    if result["status"] == "infra_error":
        raise InfraError(result.get("error") or "실행 환경 오류")
    return result


def _round(iteration: int, impl: str, test_src: str | None, result: dict) -> dict:
    return {
        "iteration": iteration, "code": impl, "smoke_test": test_src, "static": static_check(impl),
        "exec": result, "coder_error": None, "feedback_source": None,
        "reviewed": False, "review_passed": None, "review_feedback": "", "reviewer_error": None,
    }


async def run_arm(
    arm: str, state: dict, first_code: str, first_exec: dict, shared_review: dict | None,
    run_exec: Callable[..., Awaitable[dict]], on_round: Callable[[list[dict]], None],
) -> dict:
    """같은 최초 코드에서 한 조건을 끝까지 돌린다. 회차가 끝날 때마다 on_round로 저장할 기회를 준다."""
    rounds: list[dict] = []
    impl, fixed_test = exec_check.split_smoke_test(first_code)
    result = first_exec
    st = {**state, "generated_code": _combined(impl, fixed_test), "iteration_count": 1}
    # 분기 이후의 호출 수 — 최초 Coder와 공통 1회차 Reviewer는 run_paper의 first.calls에 따로 센다
    coder_calls, reviewer_calls = 0, 0
    feedback = ""
    last_review = ""  # 아직 해결되지 않았을 수 있는 Reviewer 지적 — 실행 피드백에 덮이지 않게 함께 넘긴다
    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            if iteration > 1:
                st["review_feedback"] = feedback
                update = await coder.coder_node(st)
                coder_calls += 1
                coder_error = update.get("error")
                if coder_error:
                    raise LLMError(f"Coder: {coder_error}")
                st.update(update)
                new_impl, new_test = exec_check.split_smoke_test(st.get("generated_code", ""))
                if fixed_test is None:
                    fixed_test = new_test  # 아직 고정된 테스트가 없을 때만 받아들인다
                impl = new_impl
                st["generated_code"] = _combined(impl, fixed_test)
                result = await _checked(impl, fixed_test, run_exec)
            rnd = _round(iteration, impl, fixed_test, result)
            if iteration > 1:
                rnd["coder_error"] = coder_error
            rounds.append(rnd)

            if arm == "exec" and result["status"] in exec_check.CODE_ISSUES:
                feedback = exec_check.exec_feedback(result, fixed_test)
                if last_review:
                    feedback += "\n\n### 이전 Reviewer 지적 (아직 반영되지 않았을 수 있다)\n" + last_review
                rnd["feedback_source"] = "exec"
                on_round(rounds)
                continue

            if iteration == 1 and shared_review is not None:
                review = shared_review
            else:
                review = await reviewer.reviewer_node(st)
                reviewer_calls += 1
            rnd.update(reviewed=True, review_passed=review.get("review_passed", False),
                       review_feedback=review.get("review_feedback", ""), reviewer_error=review.get("error"))
            on_round(rounds)
            if review.get("error"):
                raise LLMError(f"Reviewer: {review['error']}")
            if rnd["review_passed"]:
                return {"end_reason": "passed", "rounds": rounds, "run_error": None,
                        "coder_calls": coder_calls, "reviewer_calls": reviewer_calls}
            feedback = last_review = review.get("review_feedback", "")
            # 조건에 따라 같은 피드백 경로를 쓴다 — 고정된 테스트를 알려 이름이 바뀌지 않게 한다
            if fixed_test:
                feedback += ("\n\n### 고정된 smoke_test — 수정하지 말 것. 이 테스트가 호출하는 이름·인자를 유지하라.\n```python\n"
                             + fixed_test.strip() + "\n```")
            rnd["feedback_source"] = "reviewer"
        return {"end_reason": "max_iterations", "rounds": rounds, "run_error": None,
                "coder_calls": coder_calls, "reviewer_calls": reviewer_calls}
    except InfraError as e:
        return {"end_reason": "error", "rounds": rounds, "run_error": f"실행 환경 오류: {e}",
                "coder_calls": coder_calls, "reviewer_calls": reviewer_calls}
    except LLMError as e:
        return {"end_reason": "error", "rounds": rounds, "run_error": f"LLM 호출 오류: {e}",
                "coder_calls": coder_calls, "reviewer_calls": reviewer_calls}
    except Exception as e:
        return {"end_reason": "error", "rounds": rounds, "run_error": f"{type(e).__name__}: {e}",
                "coder_calls": coder_calls, "reviewer_calls": reviewer_calls}


async def run_paper(name: str, on_update: Callable[[dict], None],
                    run_exec: Callable[..., Awaitable[dict]] = exec_check.run_exec_check) -> dict:
    paper = PAPERS[name]
    started = time.perf_counter()
    out: dict = {"paper": name, "title": paper["title"], "pages": 0, "run_error": None, "first": None, "arms": {}}

    def save_arm(arm: str) -> Callable[[list[dict]], None]:
        def _save(rounds: list[dict]) -> None:
            out["arms"][arm] = {"end_reason": "running", "rounds": rounds}
            on_update(out)
        return _save

    try:
        async with asyncio.timeout(PAPER_TIMEOUT_SECONDS):
            pages = await download_pdf_pages(paper["pdf_url"])
            out["pages"] = len(pages)
            state = _make_initial_state("analyze", paper["title"], pdf_pages=pages)
            state["papers"] = [paper]
            state["analysis_source"] = "pdf"
            state.update(await analyzer_node(state))
            out["analyzer_error"] = state.get("error")
            first = await coder.coder_node(state)
            state.update(first)
            if first.get("code_skip_reason"):
                out["first"] = {"skip_reason": first["code_skip_reason"]}
                out["end_reason"] = "not_implementable"
                return out
            if first.get("error"):
                raise LLMError(f"Coder: {first['error']}")
            impl, test_src = exec_check.split_smoke_test(first.get("generated_code", ""))
            first_exec = await _checked(impl, test_src, run_exec)
            out["first"] = {"coder_error": first.get("error"), "exec": first_exec,
                            "calls": {"coder": 1, "reviewer": 0}}
            on_update(out)
            # 두 조건의 1회차 Reviewer가 같은 입력이면 한 번만 부른다
            shared_review = None
            if first_exec["status"] == "passed":
                shared_review = await reviewer.reviewer_node(
                    {**state, "generated_code": _combined(impl, test_src), "iteration_count": 1})
                out["first"]["calls"]["reviewer"] = 1
            for arm in ARMS:
                out["arms"][arm] = await run_arm(arm, state, first.get("generated_code", ""), first_exec,
                                                 shared_review, run_exec, save_arm(arm))
                on_update(out)
    except Exception as e:
        out["run_error"] = f"{type(e).__name__}: {e}"
        # 도중에 멈춘 조건은 이전 회차의 통과로 집계되지 않게 오류로 닫는다
        for a in out["arms"].values():
            if a.get("end_reason") == "running":
                a.update(end_reason="error", run_error=out["run_error"])
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


def summarize(results: list[dict]) -> dict:
    """조건별 1회차·최종 실행 통과 수를 센다. 실행 검사가 없는(생략·오류) 논문은 따로 센다."""
    summary: dict = {}
    for arm in ARMS:
        first_pass = final_pass = measured = 0
        for r in results:
            a = r.get("arms", {}).get(arm)
            # 끝까지 간 조건만 센다 (통과 또는 횟수 상한) — 오류·중단은 최종 결과가 없다
            if not a or not a.get("rounds") or a.get("end_reason") not in ("passed", "max_iterations"):
                continue
            measured += 1
            first_pass += a["rounds"][0]["exec"]["status"] == "passed"
            final_pass += a["rounds"][-1]["exec"]["status"] == "passed"
        summary[arm] = {"measured": measured, "first_exec_passed": first_pass, "final_exec_passed": final_pass}
    summary["not_measured"] = [r["paper"] for r in results if not r.get("arms")]
    # 테스트 형식 위반이 한 번이라도 나온 논문 — 그 회차는 실행 검사로 판단하지 못했다
    summary["test_error_papers"] = sorted({r["paper"] for r in results for a in r.get("arms", {}).values()
                                           for x in a.get("rounds", []) if x["exec"]["status"] == "test_error"})
    return summary


def report(results: list[dict]) -> None:
    for r in results:
        print(f"\n{r['paper']}: {r.get('end_reason', '')} {r.get('run_error') or ''}")
        for arm, a in r.get("arms", {}).items():
            steps = " → ".join(
                f"{x['exec']['status']}{'/검토통과' if x['review_passed'] else '/낙제' if x['reviewed'] else ''}" for x in a["rounds"])
            print(f"  {arm}: {a['end_reason']} | {steps} {a.get('run_error') or ''}")
    print("\n" + json.dumps(summarize(results), ensure_ascii=False))


async def run(names: list[str], out_path: str | None) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    image = subprocess.run(["docker", "image", "inspect", exec_check.IMAGE, "--format", "{{.Id}}"],
                           capture_output=True, text=True).stdout.strip()
    meta = {
        "commit": commit, "image": exec_check.IMAGE, "image_id": image,
        "limits": {"time_s": exec_check.TIME_LIMIT_SECONDS, "host_timeout_s": exec_check.HOST_TIMEOUT_SECONDS,
                   "memory": exec_check.MEMORY_LIMIT, "cpus": exec_check.CPU_LIMIT, "network": "none"},
        "models": {"coder": coder._MODEL, "reviewer": reviewer._MODEL}, "smoke_rules": SMOKE_RULES,
    }
    results: list[dict] = []

    def save(_: dict | None = None) -> None:
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "results": results, "summary": summarize(results)}, f, ensure_ascii=False, indent=2)

    for name in names:
        print(f"{name} 실행 중...", flush=True)
        current: dict = {}
        results.append(current)

        def on_update(out: dict, _current: dict = current) -> None:
            _current.clear()
            _current.update(out)
            save()

        on_update(await run_paper(name, on_update))
    report(results)
    if out_path:
        print(f"\n저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", choices=list(PAPERS), default=MAIN_PAPERS)
    parser.add_argument("--out", help="결과(회차별 코드·실행 결과 포함)를 JSON으로 저장할 경로")
    args = parser.parse_args()
    asyncio.run(run(args.papers, args.out))
