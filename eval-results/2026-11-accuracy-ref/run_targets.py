"""사전 등록(SPEC.txt)대로 대상 코드 6개를 기준 계산과 비교한다. Docker 안에서 네트워크 없이 돌린다.

실행 (저장소 루트에서):
  docker run --rm --network none --read-only --tmpfs /tmp --user 65534:65534 --memory 2g --cpus 2 \
    -v "$PWD/eval-results/2026-11-accuracy-ref:/work:ro" --entrypoint python paperpilot-exec-check:dev -I /work/run_targets.py
결과는 표준 출력의 JSON 한 줄(RESULT:)로 낸다 — 읽기 전용 마운트라 파일을 쓰지 않는다.
"""

import json
import sys
import traceback

sys.path.insert(0, "/work")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import adapters  # noqa: E402
import reference as ref  # noqa: E402

TARGETS = [(p, v) for p in ("rotated_lora_optimizer", "avq_attention", "explicit_dropout") for v in ("first", "exec_final")]


def load(paper: str, version: str) -> dict:
    # #60 runner처럼 모듈로 등록한다 — @dataclass가 sys.modules에서 자기 모듈을 찾는다 (1차 실행에서 등록하지 않아 4건이 불러오기 오류였다)
    import types

    module = types.ModuleType(f"target_{paper}_{version}")  # 이름이 __main__이 아니어서 __main__ 블록은 돌지 않는다
    sys.modules[module.__name__] = module
    src = open(f"/work/targets/{paper}__{version}.py").read()
    exec(compile(src, f"{paper}__{version}.py", "exec"), module.__dict__)
    return module.__dict__


def close(got, want) -> bool:
    return all(np.asarray(g).shape == np.asarray(w).shape and np.allclose(g, w, rtol=ref.RTOL, atol=ref.ATOL)
               for g, w in zip(got, want))


def check(paper: str, ns: dict) -> list[dict]:
    """입력별 (이름, 일치 여부, 대상 값, 기대값)"""
    rows = []
    if paper == "rotated_lora_optimizer":
        h = ref.RLORA_HAND
        r = ref.rlora_random()
        for name, inp, want in (("손계산 입력", h, (h["S"], h["Up"], h["Vp"])),
                                ("무작위 직교 U·V (seed 0)", r, ref.rlora_components(r["G"], r["U"], r["V"]))):
            got = adapters.rlora(ns, torch, inp["G"], inp["U"], inp["V"])
            rows.append({"input": name, "match": close(got, want),
                         "got": [g.round(6).tolist() for g in got], "want": [w.round(6).tolist() for w in want]})
    elif paper == "avq_attention":
        a = ref.AVQ_HAND
        got = adapters.avq(ns, torch, a["K"], a["V"], a["C"], a["M0"], a["n_children"])
        want = (a["Vbar"], a["n"])
        rows.append({"input": "손계산 입력", "match": close(got, want),
                     "got": [g.round(6).tolist() for g in got], "want": [w.round(6).tolist() for w in want]})
    else:
        value, layers = adapters.dropout(ns, torch, p=0.3, tokens=4, d=3, h=5, seed=0)
        per_layer = [ref.ffn_regularizer(Z, W, 0.3) for Z, W in layers]
        want = sum(per_layer)
        rows.append({"input": "배치 1, 토큰 4, d=3, h=5, p=0.3, seed 0 (J₁+J₂)", "match": close([value], [want]),
                     "got": value, "want": want, "want_per_layer": per_layer,
                     "ratio_got_over_want": value / want if want else None,
                     "layer_inputs_shape": [list(Z.shape) for Z, _ in layers]})
    return rows


def main() -> None:
    results = []
    for paper, version in TARGETS:
        item = {"paper": paper, "version": version}
        try:
            ns = load(paper, version)
        except Exception as e:  # 대상 코드를 불러오다 예외 — 실행 불가
            item.update(result="대상 코드 실행 불가", error=f"불러오기: {type(e).__name__}: {e}")
            results.append(item)
            continue
        try:
            rows = check(paper, ns)
            item["rows"] = rows
            item["result"] = "수치 일치" if all(r["match"] for r in rows) else "수치 불일치"
        except adapters.AdapterError as e:
            item.update(result="평가 측 문제로 판단 불가", error=str(e))
        except Exception as e:
            item.update(result="대상 코드 실행 불가", error="".join(traceback.format_exception_only(type(e), e)).strip(),
                        where=traceback.format_exc()[-800:])
        results.append(item)
    print("RESULT:" + json.dumps({"torch": torch.__version__, "numpy": np.__version__, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
