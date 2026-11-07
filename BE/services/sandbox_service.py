"""생성 코드 실행 검사 — E2B 샌드박스에서 구현 코드 + 고정된 smoke_test를 실행한다.

운영 서버(1GB)에서는 torch 실행(RSS 약 320MB)을 감당하지 못해 외부 샌드박스를 쓴다.
- 샌드박스마다 인터넷을 끈다. E2B 키는 샌드박스에 넘기지 않는다 — 코드·테스트·runner만 올린다
- 샌드박스 자체 수명을 짧게 잡아, 취소 등으로 우리가 지우지 못해도 E2B가 끝낸다
- 실행 환경 문제는 "unavailable"로 돌려준다 (코드 탓이 아니다). 인증 실패·결제/한도 문제는 disable=True —
  같은 분석에서 다시 시도하지 않는다
"""

import asyncio
import logging
import uuid
from pathlib import Path

from e2b import AsyncSandbox, CommandExitException
from e2b.exceptions import AuthenticationException, TimeoutException

from agents import exec_check as common
from core.config import settings

logger = logging.getLogger(__name__)

RUNNER_SOURCE = (Path(__file__).resolve().parent.parent / "sandbox" / "runner.py").read_text()
WORK_DIR = "/home/user/work"
TIME_LIMIT_SECONDS = 100          # runner 안의 제한 (import부터 검사까지)
COMMAND_TIMEOUT_SECONDS = 130     # 명령 연결 제한 — runner 제한 + torch 로딩 여유
SANDBOX_LIFETIME_SECONDS = 180    # 샌드박스 자체 수명. 지우지 못한 샌드박스도 이 시간 뒤 E2B가 끝낸다
REQUEST_TIMEOUT_SECONDS = 30      # 생성·업로드·삭제 같은 API 호출 하나의 제한
SLOT_WAIT_SECONDS = 10            # 동시 실행 자리를 기다리는 시간
OUTPUT_CAP_CHARS = 64 * 1024      # 받아 두는 출력 상한 — 생성 코드가 출력을 쏟아내도 메모리를 채우지 않게
LIMITS = f"(제한: 실행 {TIME_LIMIT_SECONDS}초, 메모리 2GB, CPU 2개, 네트워크 없음)"
# 같은 분석에서 다시 시도해도 소용없는 응답 — 인증 실패, 결제·사용 한도
_DISABLING_STATUS = (401, 402, 403)

# 프로세스 안의 동시 실행 상한. 배포 전환 중 두 인스턴스·수동 실험은 각자 따로 센다 (E2B 계정 한도는 동시 20)
_slots = asyncio.Semaphore(settings.exec_check_max_concurrent)


def unavailable(reason: str, disable: bool = False) -> dict:
    return {"status": "unavailable", "error": reason, "disable": disable}


def _api_failure(stage: str, e: BaseException) -> dict:
    status = getattr(e, "status_code", None)
    disable = isinstance(e, AuthenticationException) or status in _DISABLING_STATUS
    logger.warning(f"[ExecCheck] {stage} 실패 ({type(e).__name__}, status={status}): {e}")
    return unavailable(f"실행 환경 오류 ({stage}): {type(e).__name__}", disable=disable)


class _CappedOutput:
    """끝부분만 남기는 출력 버퍼 — 결과 줄은 마지막에 찍히므로 끝을 남긴다."""

    def __init__(self) -> None:
        self.text = ""

    def add(self, chunk: str) -> None:
        self.text = (self.text + chunk)[-OUTPUT_CAP_CHARS:]


def classify(stdout: str, exit_code: int | None, timed_out: bool, nonce: str) -> dict:
    """E2B 실행의 관측값을 결과 상태로 바꾼다. runner 결과 해석은 평가(Docker)와 공유한다.
    메모리 초과(12GB 할당)는 E2B에서 결과 줄 없이 종료 코드 -1로 보였다 (2026-11-07, 1회 관측). -1이 메모리 초과만 뜻하는지는
    모르므로 oom으로 단정하지 않고 code_error + 사유로 둔다 — 어느 쪽이든 Coder에게 돌아가는 경로는 같다.
    """
    if timed_out:
        return {"status": "timeout", "error": f"{COMMAND_TIMEOUT_SECONDS}초 안에 끝나지 않아 중단했습니다"}
    found, runner = common.find_runner_result(stdout, nonce)
    if not found:
        hint = " — 강제 종료로 보입니다. 메모리 2GB 초과일 수 있습니다" if exit_code in (-1, 137, -9) else ""
        return {"status": "code_error",
                "error": f"검사 결과 없이 종료됐습니다 (종료 코드 {exit_code}){hint} — 생성 코드가 프로세스를 끝냈을 수 있습니다"}
    if runner is None:
        return {"status": "code_error", "error": "검사 결과의 형식이 올바르지 않습니다"}
    if exit_code != 0:
        return {"status": "code_error", "error": f"결과 출력 뒤 비정상 종료 (종료 코드 {exit_code})", "runner": runner}
    return common.interpret_runner(runner)


async def run_exec_check(impl: str, test_src: str | None) -> dict:
    """구현 코드 + 고정된 smoke_test를 E2B에서 실행한다. 취소(CancelledError)는 그대로 올려 보낸다."""
    if not test_src:
        return common.no_test_result()
    if not settings.e2b_api_key:
        return unavailable("실행 검사가 설정되지 않았습니다", disable=True)
    try:
        await asyncio.wait_for(_slots.acquire(), SLOT_WAIT_SECONDS)
    except asyncio.TimeoutError:
        return unavailable("동시에 진행 중인 실행 검사가 많아 이번에는 건너뛰었습니다")
    try:
        return await _run(impl, test_src)
    finally:
        _slots.release()


async def _kill(sandbox: AsyncSandbox) -> None:
    try:
        await asyncio.wait_for(sandbox.kill(request_timeout=REQUEST_TIMEOUT_SECONDS), REQUEST_TIMEOUT_SECONDS + 5)
    except Exception as e:
        # 지우지 못해도 샌드박스 수명(SANDBOX_LIFETIME_SECONDS)이 끝나면 E2B가 정리한다
        logger.warning(f"[ExecCheck] 샌드박스 삭제 실패 (수명 {SANDBOX_LIFETIME_SECONDS}초 뒤 자동 종료): {e}")


async def _run(impl: str, test_src: str) -> dict:
    try:
        sandbox = await asyncio.wait_for(AsyncSandbox.create(
            template=settings.exec_check_template,
            timeout=SANDBOX_LIFETIME_SECONDS,
            allow_internet_access=False,
            api_key=settings.e2b_api_key,
            request_timeout=REQUEST_TIMEOUT_SECONDS,
        ), REQUEST_TIMEOUT_SECONDS + 5)
    except Exception as e:  # asyncio.TimeoutError도 Exception이다. 취소(CancelledError)는 잡지 않는다
        return _api_failure("샌드박스 생성", e)
    try:
        return await _execute(sandbox, impl, test_src)
    finally:
        # 취소된 뒤에도 삭제 요청은 끝까지 보낸다
        await asyncio.shield(_kill(sandbox))


async def _execute(sandbox: AsyncSandbox, impl: str, test_src: str) -> dict:
    try:
        await sandbox.files.write_files([
            {"path": f"{WORK_DIR}/runner.py", "data": RUNNER_SOURCE},
            {"path": f"{WORK_DIR}/generated.py", "data": impl},
            {"path": f"{WORK_DIR}/smoke_test.py", "data": test_src},
        ], request_timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        return _api_failure("파일 업로드", e)

    nonce = uuid.uuid4().hex
    output = _CappedOutput()
    try:
        handle = await sandbox.commands.run(
            f"python -I {WORK_DIR}/runner.py {WORK_DIR}",
            background=True, stdin=True, cwd=WORK_DIR,
            envs={"EXEC_TIME_LIMIT": str(TIME_LIMIT_SECONDS)},
            on_stdout=output.add, on_stderr=output.add,
            timeout=COMMAND_TIMEOUT_SECONDS, request_timeout=REQUEST_TIMEOUT_SECONDS,
            retain_output=False,
        )
        # nonce는 stdin으로만 준다 — 파일·환경 변수에 두면 생성 코드가 읽어 가짜 결과 줄을 만들 수 있다
        await sandbox.commands.send_stdin(handle.pid, nonce + "\n", request_timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        return _api_failure("실행 시작", e)

    exit_code: int | None = None
    timed_out = False
    try:
        result = await asyncio.wait_for(handle.wait(), COMMAND_TIMEOUT_SECONDS + 10)
        exit_code = result.exit_code
    except CommandExitException as e:
        exit_code = e.exit_code
    except (TimeoutException, asyncio.TimeoutError):
        timed_out = True
    except Exception as e:
        return _api_failure("실행 대기", e)
    return classify(output.text, exit_code, timed_out, nonce)
