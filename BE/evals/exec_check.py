"""생성 코드 실행 검사 (평가 전용). 격리된 Docker 컨테이너에서 구현 코드와 고정된 smoke_test를 실행한다.

결과 상태:
- passed      — smoke_test가 끝까지 돌고 반환값 검사(빈 값·NaN·gradient)를 통과
- code_error  — 코드 문제(예외, 반환값 위반, smoke_test 없음, 비정상 종료). Coder에게 돌려줄 수 있다
- timeout     — 제한 시간 안에 끝나지 않음
- oom         — 메모리 제한을 넘어 강제 종료
- test_error  — smoke_test의 반환 형식 위반 (테스트 쪽 결함). 구현을 고쳐도 해결되지 않으므로 Coder에게 보내지 않는다
- infra_error — Docker·이미지 등 실행 환경 문제. 코드 탓이 아니므로 Coder에게 보내지 않는다
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from agents import exec_check as common
from agents.exec_check import CODE_ISSUES, RESULT_PREFIX, no_test_result, split_smoke_test  # noqa: F401 — 평가 코드가 이 이름으로 쓴다

IMAGE = "paperpilot-exec-check:dev"
TIME_LIMIT_SECONDS = 100        # 컨테이너 안 runner의 제한 (import부터 검사까지)
HOST_TIMEOUT_SECONDS = 130      # 호스트가 컨테이너를 강제 종료하는 시각 — runner 제한 + 기동 여유
MEMORY_LIMIT = "2g"
CPU_LIMIT = "2"
STDOUT_CAP_BYTES = 64 * 1024    # 호스트가 읽어 들이는 출력 상한 — 읽는 도중에 자른다


def classify(stdout: str, exit_code: int | None, oom_killed: bool, timed_out: bool, launch_error: str | None,
             nonce: str = "") -> dict:
    """Docker 실행의 관측값을 결과 상태로 바꾼다 (Docker 없이 테스트할 수 있게 분리).
    종료 코드 125~127·OOMKilled는 Docker에만 있는 관측이다 — runner 결과 해석만 운영과 공유한다.
    """
    if launch_error:
        return {"status": "infra_error", "error": launch_error}
    if timed_out:
        return {"status": "timeout", "error": f"{HOST_TIMEOUT_SECONDS}초 안에 컨테이너가 끝나지 않아 강제 종료했습니다"}
    if oom_killed:
        return {"status": "oom", "error": f"메모리 제한 {MEMORY_LIMIT}을 넘어 강제 종료됐습니다"}
    if exit_code in (125, 126, 127):
        return {"status": "infra_error", "error": f"docker run 실패 (종료 코드 {exit_code}): {stdout[-500:]}"}
    found, runner = common.find_runner_result(stdout, nonce)
    if not found:
        return {"status": "code_error", "error": f"검사 결과 없이 종료됐습니다 (종료 코드 {exit_code}) — 생성 코드가 프로세스를 끝냈을 수 있습니다"}
    if runner is None:
        return {"status": "code_error", "error": "검사 결과의 형식이 올바르지 않습니다"}
    if exit_code != 0:
        return {"status": "code_error", "error": f"결과 출력 뒤 비정상 종료 (종료 코드 {exit_code})", "runner": runner}
    return common.interpret_runner(runner)


DOCKER_AUX_TIMEOUT_SECONDS = 30  # kill·inspect·rm — 데몬이 응답하지 않아도 무기한 기다리지 않는다


async def _docker(*args: str) -> tuple[int, str]:
    """docker 보조 명령. 실행 파일이 없거나 시간이 넘으면 (-1, 사유)를 돌려준다 — 예외로 정리를 끊지 않는다."""
    try:
        proc = await asyncio.create_subprocess_exec("docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    except OSError as e:
        return -1, f"docker 실행 실패: {e}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), DOCKER_AUX_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, f"docker {args[0]}가 {DOCKER_AUX_TIMEOUT_SECONDS}초 안에 끝나지 않았습니다"
    return proc.returncode or 0, out.decode(errors="replace")


async def _read_capped(stream: asyncio.StreamReader) -> str:
    """상한까지만 모으고 나머지는 읽어서 버린다 — 파이프가 막혀 컨테이너가 멈추지 않게."""
    kept = bytearray()
    while chunk := await stream.read(8192):
        if len(kept) < STDOUT_CAP_BYTES:
            kept += chunk[: STDOUT_CAP_BYTES - len(kept)]
    return kept.decode(errors="replace")


async def run_exec_check(impl: str, test_src: str | None) -> dict:
    """구현 코드 + 고정된 smoke_test를 격리 컨테이너에서 실행한다. 취소·예외에도 컨테이너와 임시 폴더를 지운다.
    컨테이너를 지우지 못하면 결과 대신 infra_error를 돌려준다 — 남은 컨테이너를 모른 채 다음 평가로 넘어가지 않게.
    """
    if not test_src:
        return no_test_result()
    workdir = Path(tempfile.mkdtemp(prefix="exec-check-"))
    name = f"exec-check-{uuid.uuid4().hex[:12]}"
    nonce = uuid.uuid4().hex
    launched = {"started": False}
    cleanup_error = None
    try:
        result = await _execute(impl, test_src, workdir, name, nonce, launched)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if launched["started"]:
            code, out = await asyncio.shield(_docker("rm", "-f", name))
            if code != 0 and "No such container" not in out:
                cleanup_error = f"컨테이너 {name} 정리 실패: {out[-200:]}"
    if cleanup_error:
        return {"status": "infra_error", "error": cleanup_error, "before_cleanup": result}
    return result


async def _execute(impl: str, test_src: str, workdir: Path, name: str, nonce: str, launched: dict) -> dict:
    (workdir / "generated.py").write_text(impl)
    (workdir / "smoke_test.py").write_text(test_src)
    os.chmod(workdir, 0o755)
    for f in workdir.iterdir():
        os.chmod(f, 0o644)
    args = [
        "run", "-i", "--name", name,
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", MEMORY_LIMIT, "--memory-swap", MEMORY_LIMIT,  # 스왑까지 같은 한도
        "--cpus", CPU_LIMIT, "--pids-limit", "128",
        "--user", "65534:65534", "-e", f"EXEC_TIME_LIMIT={TIME_LIMIT_SECONDS}",
        "-v", f"{workdir}:/work:ro", IMAGE,
    ]
    try:
        proc = await asyncio.create_subprocess_exec("docker", *args, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    except OSError as e:
        return classify("", None, False, False, f"docker를 실행하지 못했습니다: {e}")
    launched["started"] = True
    proc.stdin.write((nonce + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()
    timed_out = False
    reader = asyncio.ensure_future(_read_capped(proc.stdout))
    try:
        await asyncio.wait_for(proc.wait(), HOST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        timed_out = True
        await _docker("kill", name)
        try:
            await asyncio.wait_for(proc.wait(), DOCKER_AUX_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    stdout = await asyncio.wait_for(reader, DOCKER_AUX_TIMEOUT_SECONDS)
    code, state = await _docker("inspect", name, "--format", "{{.State.OOMKilled}}")
    oom = code == 0 and state.strip() == "true"
    if code != 0 and proc.returncode not in (125, 126, 127):
        return classify("", None, False, False, f"컨테이너 상태를 확인하지 못했습니다: {state[-300:]}")
    return classify(stdout, proc.returncode, oom, timed_out, None, nonce)


def exec_feedback(result: dict, test_src: str | None) -> str:
    return common.exec_feedback(result, test_src, f"(제한: 실행 {TIME_LIMIT_SECONDS}초, 메모리 {MEMORY_LIMIT}, CPU {CPU_LIMIT}개, 네트워크 없음)")
