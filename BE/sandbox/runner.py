"""컨테이너 안에서 생성 코드와 고정된 smoke_test를 실행하고 결과를 JSON 한 줄로 출력한다.

실행: python -I runner.py [작업 폴더] (기본 /work — 평가용 Docker. 운영 E2B는 /home/user/work를 넘긴다)
<작업 폴더>/generated.py — 구현 코드 (smoke_test 제외)
<작업 폴더>/smoke_test.py — 고정된 smoke_test 함수 소스
결과는 표준 출력의 마지막 줄에 RESULT_PREFIX로 시작하는 JSON으로 쓴다. 생성 코드의 출력은 버퍼에 가둔다.
생성 코드와 같은 프로세스라 완전한 신뢰 경계는 아니다 — 호스트는 종료 코드·결과 형식을 함께 확인한다.
"""

import io
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
import types

RESULT_PREFIX = "__EXEC_CHECK_RESULT__"
TIME_LIMIT_SECONDS = int(os.environ.get("EXEC_TIME_LIMIT", "100"))
OUTPUT_KEEP_CHARS = 4000


class _CappedBuffer(io.TextIOBase):
    """생성 코드의 출력은 끝부분만 남긴다 — 무한 출력이 메모리를 채우지 않게."""

    def __init__(self) -> None:
        self.tail = ""

    def write(self, s: str) -> int:
        self.tail = (self.tail + s)[-OUTPUT_KEEP_CHARS:]
        return len(s)

    def flush(self) -> None:
        pass


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ANN001
    raise _Timeout(f"{TIME_LIMIT_SECONDS}초 안에 끝나지 않았습니다")


def _check_returned(value, torch) -> tuple[list[str], list[str]]:  # noqa: ANN001
    """smoke_test의 반환값을 검사한다. (코드 결과의 문제, 테스트 형식 위반)을 따로 돌려준다.
    형식 위반은 테스트 쪽 결함이라 구현을 고쳐도 해결되지 않는다 — 호스트가 Coder에게 보내지 않는다.
    """
    if not isinstance(value, dict) or "outputs" not in value or "grad_params" not in value:
        return [], ['smoke_test는 {"outputs": [텐서...], "grad_params": [파라미터...]} 형태의 dict를 반환해야 합니다']
    problems: list[str] = []
    test_problems: list[str] = []
    outputs, grad_params = value["outputs"], value["grad_params"]
    if not isinstance(outputs, (list, tuple)) or not outputs:
        test_problems.append("outputs는 텐서가 하나 이상 든 리스트여야 합니다")
        outputs = []
    for i, t in enumerate(outputs):
        if not isinstance(t, torch.Tensor):
            test_problems.append(f"outputs[{i}]가 텐서가 아닙니다 ({type(t).__name__})")
        elif t.numel() == 0:
            problems.append(f"outputs[{i}]가 빈 텐서입니다 (shape {tuple(t.shape)})")
        elif t.is_floating_point() or t.is_complex():
            if not bool(torch.isfinite(t.detach()).all()):
                problems.append(f"outputs[{i}]에 NaN 또는 inf가 있습니다 (shape {tuple(t.shape)})")
    if not isinstance(grad_params, (list, tuple)):
        test_problems.append("grad_params는 리스트여야 합니다 (학습할 파라미터가 없으면 빈 리스트)")
        grad_params = []
    for i, p in enumerate(grad_params):
        if not isinstance(p, torch.Tensor):
            test_problems.append(f"grad_params[{i}]가 텐서가 아닙니다 ({type(p).__name__})")
        elif not (p.is_leaf and p.requires_grad):
            # 흔한 실수: param.grad.clone()처럼 gradient 텐서를 넣는다 — 파라미터 자체를 넣어야 한다
            test_problems.append(f"grad_params[{i}]가 학습 파라미터(requires_grad인 leaf 텐서)가 아닙니다 — gradient가 아니라 파라미터 자체를 넣어야 합니다")
        elif p.grad is None:
            problems.append(f"grad_params[{i}]의 gradient가 없습니다 — backward가 이 파라미터까지 닿지 않았습니다 (shape {tuple(p.shape)})")
        elif not bool(torch.isfinite(p.grad).all()):
            problems.append(f"grad_params[{i}]의 gradient에 NaN 또는 inf가 있습니다")
    return problems, test_problems


def main() -> None:
    # 생성 코드를 실행하기 전에 호스트가 준 일회용 값을 읽는다 — 생성 코드가 찍은 가짜 결과 줄과 구분한다
    nonce = sys.stdin.readline().strip()
    work = sys.argv[1] if len(sys.argv) > 1 else "/work"
    result: dict = {"passed": False, "stage": "start", "error": None, "problems": [], "test_problems": []}
    captured = _CappedBuffer()
    real_stdout = sys.stdout
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        import torch

        torch.manual_seed(0)
        result["torch_version"] = torch.__version__
        sys.stdout = sys.stderr = captured
        module = types.ModuleType("generated")
        module.__file__ = os.path.join(work, "generated.py")
        sys.modules["generated"] = module
        result["stage"] = "import"
        exec(compile(open(os.path.join(work, "generated.py")).read(), "generated.py", "exec"), module.__dict__)
        result["stage"] = "define_test"
        exec(compile(open(os.path.join(work, "smoke_test.py")).read(), "smoke_test.py", "exec"), module.__dict__)
        if not callable(module.__dict__.get("smoke_test")):
            raise RuntimeError("smoke_test 함수가 정의되지 않았습니다")
        result["stage"] = "run_test"
        value = module.__dict__["smoke_test"]()
        result["stage"] = "check"
        result["problems"], result["test_problems"] = _check_returned(value, torch)
        result["passed"] = not result["problems"] and not result["test_problems"]
        result["stage"] = "done"
    except _Timeout as e:
        result["error"] = f"Timeout: {e}"
        result["timeout"] = True
    except BaseException as e:  # noqa: BLE001 — SystemExit 등도 실패로 기록한다
        result["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-OUTPUT_KEEP_CHARS:]
    finally:
        signal.alarm(0)
        sys.stdout = sys.stderr = real_stdout
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # 리눅스는 KB
    result["output_tail"] = captured.tail
    real_stdout.write("\n" + RESULT_PREFIX + nonce + ":" + json.dumps(result, ensure_ascii=False) + "\n")
    real_stdout.flush()
    os._exit(0)  # 생성 코드가 남긴 스레드·atexit 훅이 종료를 막지 않게


if __name__ == "__main__":
    main()
