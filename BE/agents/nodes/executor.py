import logging

from agents import exec_check
from agents.log_stream import emit_log
from agents.state import AgentState
from services import sandbox_service

logger = logging.getLogger(__name__)


async def executor_node(state: AgentState) -> dict:
    """Coder가 쓴 코드와 smoke_test를 격리된 샌드박스(E2B)에서 실행한다.

    smoke_test는 처음 나온 회차의 것으로 고정한다 — Coder가 테스트를 쉽게 고쳐 통과시키지 못하게.
    실행 오류(code_error·timeout·oom)는 Coder에게 돌려줄 피드백을 만든다. 피드백에는 아직 반영되지 않았을 수 있는
    직전 Reviewer 지적을 함께 넣는다. 실행 환경을 쓰지 못하면 "unavailable"로 두고 Reviewer만으로 진행한다.
    """
    iteration = state.get("iteration_count", 0)
    impl, new_test = exec_check.split_smoke_test(state.get("generated_code", ""))
    fixed_test = state.get("smoke_test") or new_test
    update: dict = {
        "smoke_test": fixed_test or "",
        "generated_code": exec_check.with_fixed_test(state.get("generated_code", ""), fixed_test),
        "exec_feedback": "",
        "exec_detail": "",
        "current_node": "executor",
    }

    if state.get("exec_disabled"):
        emit_log("executor", "실행 검사를 건너뜁니다 (실행 환경을 쓸 수 없음)")
        return {**update, "exec_status": "unavailable", "exec_detail": state.get("exec_detail", "")}

    emit_log("executor", f"{iteration}회차 코드 실행 검사 중 (외부 샌드박스, 네트워크 차단)...")
    result = await sandbox_service.run_exec_check(impl, fixed_test)
    status = result["status"]
    detail = exec_check.summary(result)
    update.update(exec_status=status, exec_detail=detail)
    logger.info(f"[Executor] {iteration}회차 — {status} {detail[-200:]!r}")

    if status == "passed":
        emit_log("executor", "✅ 실행 통과 — smoke_test가 오류 없이 끝났습니다")
    elif status in exec_check.CODE_ISSUES:
        last_line = detail.splitlines()[-1] if detail else ""
        emit_log("executor", f"❌ 실행 실패 ({status}): {last_line}")
        feedback = exec_check.exec_feedback(result, fixed_test, sandbox_service.LIMITS)
        if previous := state.get("review_feedback"):
            feedback += "\n\n### 이전 Reviewer 지적 (아직 반영되지 않았을 수 있다)\n" + previous
        update["exec_feedback"] = feedback
    elif status == "test_error":
        emit_log("executor", "smoke_test의 반환 형식이 규칙과 달라 실행 결과를 판단하지 못했습니다")
    else:
        emit_log("executor", f"실행 확인 못 함: {result.get('error') or ''}")
        if result.get("disable"):
            update["exec_disabled"] = True
    return update
