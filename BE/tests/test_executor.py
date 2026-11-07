"""운영 실행 검사(E2B) 테스트 — 가짜 샌드박스로 분기·분류·정리를 확인한다. 실제 E2B와 LLM은 부르지 않는다.

실제 E2B 통합 테스트는 RUN_E2B_TESTS=1일 때만 돈다 (과금된다 — 키가 있다는 이유만으로 돌리지 않는다).
"""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from e2b import CommandExitException
from e2b.exceptions import AuthenticationException, SandboxException
from langgraph.graph import END

from agents.exec_check import RESULT_PREFIX
from agents.nodes import executor, router
from services import agent_service, sandbox_service

IMPL = "import torch\nlin = torch.nn.Linear(4, 2)\n"
TEST = "def smoke_test():\n    return {'outputs': [lin(torch.ones(2, 4))], 'grad_params': []}\n"
CODE = IMPL + "\n\n" + TEST


def _line(nonce: str, **kw) -> str:
    base = {"passed": True, "stage": "done", "problems": [], "test_problems": [], "error": None}
    return RESULT_PREFIX + nonce + ":" + json.dumps({**base, **kw}) + "\n"


class FakeHandle:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self.pid = 7
        self.sandbox = sandbox

    async def wait(self):  # noqa: ANN201
        sb = self.sandbox
        if sb.hang:
            await asyncio.sleep(3600)
        for chunk in sb.output(sb.nonce):
            sb.on_stdout(chunk)
        if sb.exit_code != 0:
            raise CommandExitException(stdout="", stderr="", exit_code=sb.exit_code, error=None)
        return type("R", (), {"exit_code": 0})()


class FakeSandbox:
    """AsyncSandbox 대역. 시나리오는 하위 클래스 속성으로 정한다 (_scenario)."""

    create_error: BaseException | None = None
    kill_error: BaseException | None = None
    hang = False
    hang_create = False
    hang_upload = False
    exit_code = 0
    output = staticmethod(lambda nonce: [_line(nonce)])
    instances: list["FakeSandbox"] = []

    def __init__(self) -> None:
        self.killed = False
        self.written: list[dict] = []
        self.nonce = ""
        self.create_kwargs: dict = {}
        outer = self

        class _Files:
            async def write_files(self, files, request_timeout=None):  # noqa: ANN001, ANN202
                if outer.hang_upload:
                    await asyncio.sleep(3600)
                outer.written = files

        class _Commands:
            async def run(self, cmd, **kw):  # noqa: ANN001, ANN202
                outer.cmd, outer.run_kwargs = cmd, kw
                outer.on_stdout = kw["on_stdout"]
                return FakeHandle(outer)

            async def send_stdin(self, pid, data, request_timeout=None):  # noqa: ANN001, ANN202
                outer.nonce = data.strip()

        self.files, self.commands = _Files(), _Commands()

    @classmethod
    async def create(cls, **kw):  # noqa: ANN206
        if cls.create_error:
            raise cls.create_error
        if cls.hang_create:
            cls.create_started = True
            await asyncio.sleep(3600)
        sb = cls()
        sb.create_kwargs = kw
        cls.instances.append(sb)
        return sb

    async def kill(self, request_timeout=None):  # noqa: ANN001, ANN201
        self.killed = True
        if self.kill_error:
            raise self.kill_error
        return True


def _settings():  # noqa: ANN202
    s = type("S", (), {})()
    s.e2b_api_key, s.exec_check_template = "test-key", "tpl"
    return s


def _scenario(**attrs):  # noqa: ANN202
    """시나리오별 FakeSandbox 하위 클래스를 만들어 AsyncSandbox 자리에 넣는다."""
    cls = type("Scenario", (FakeSandbox,), {"instances": [], **attrs})
    return cls, patch.multiple(sandbox_service, AsyncSandbox=cls, settings=_settings())


def run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


class SandboxServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        sandbox_service._slots = asyncio.Semaphore(4)

    def test_pass_uploads_only_code_and_blocks_internet(self) -> None:
        cls, p = _scenario()
        with p:
            r = run(sandbox_service.run_exec_check(IMPL, TEST))
        self.assertEqual(r["status"], "passed")
        sb = cls.instances[0]
        self.assertTrue(sb.killed)
        self.assertIs(sb.create_kwargs["allow_internet_access"], False)
        self.assertEqual(sb.create_kwargs["timeout"], sandbox_service.SANDBOX_LIFETIME_SECONDS)
        self.assertNotIn("envs", sb.create_kwargs)  # 키를 샌드박스 환경 변수로 넘기지 않는다
        self.assertNotIn("test-key", json.dumps(sb.run_kwargs.get("envs")))
        self.assertEqual(sorted(f["path"].rsplit("/", 1)[1] for f in sb.written), ["generated.py", "runner.py", "smoke_test.py"])

    def test_forged_line_without_nonce_is_not_passed(self) -> None:
        cls, p = _scenario(output=staticmethod(lambda nonce: [_line("")]))
        with p:
            self.assertEqual(run(sandbox_service.run_exec_check(IMPL, TEST))["status"], "code_error")

    def test_runner_errors_and_exit_codes(self) -> None:
        cases = [
            (dict(output=staticmethod(lambda n: [_line(n, passed=False, stage="run_test", error="RuntimeError: shape")])), "code_error"),
            (dict(output=staticmethod(lambda n: [_line(n, passed=False, test_problems=["x"])])), "test_error"),
            (dict(output=staticmethod(lambda n: [_line(n, passed=False, stage="run_test", timeout=True)])), "timeout"),
            (dict(output=staticmethod(lambda n: [_line(n)]), exit_code=1), "code_error"),
            (dict(output=staticmethod(lambda n: ["killed\n"]), exit_code=137), "code_error"),
        ]
        for attrs, expected in cases:
            cls, p = _scenario(**attrs)
            with p:
                r = run(sandbox_service.run_exec_check(IMPL, TEST))
            self.assertEqual(r["status"], expected, attrs)
            self.assertTrue(cls.instances[0].killed)
        self.assertIn("메모리", r["error"])  # 137 — 원인은 확정하지 못하므로 사유로만 적는다

    def test_output_flood_is_capped_and_result_still_read(self) -> None:
        flood = staticmethod(lambda n: ["x" * 10_000 + "\n"] * 200 + [_line(n)])
        cls, p = _scenario(output=flood)
        with p:
            self.assertEqual(run(sandbox_service.run_exec_check(IMPL, TEST))["status"], "passed")
        buf = sandbox_service._CappedOutput()
        for chunk in flood.__func__("n"):
            buf.add(chunk)
        self.assertLessEqual(len(buf.text), sandbox_service.OUTPUT_CAP_CHARS)

    def test_auth_and_quota_failures_disable(self) -> None:
        for err, disable in [(AuthenticationException("bad key"), True), (SandboxException("pay", status_code=402), True),
                             (SandboxException("busy", status_code=500), False)]:
            cls, p = _scenario(create_error=err)
            with p:
                r = run(sandbox_service.run_exec_check(IMPL, TEST))
            self.assertEqual((r["status"], r["disable"]), ("unavailable", disable), err)

    def test_no_key_and_no_test(self) -> None:
        s = _settings()
        s.e2b_api_key = ""
        with patch.object(sandbox_service, "settings", s):
            r = run(sandbox_service.run_exec_check(IMPL, TEST))
        self.assertEqual((r["status"], r["disable"]), ("unavailable", True))
        self.assertEqual(run(sandbox_service.run_exec_check(IMPL, None))["status"], "code_error")

    def test_kill_failure_keeps_result(self) -> None:
        cls, p = _scenario(kill_error=SandboxException("gone"))
        with p:
            self.assertEqual(run(sandbox_service.run_exec_check(IMPL, TEST))["status"], "passed")
        self.assertTrue(cls.instances[0].killed)

    def _cancel_midway(self, **attrs) -> type:
        cls, p = _scenario(**attrs)

        async def go() -> None:
            task = asyncio.ensure_future(sandbox_service.run_exec_check(IMPL, TEST))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with p:
            run(go())
        self.assertEqual(sandbox_service._slots._value, 4)  # 자리를 모두 돌려줬다
        return cls

    def test_cancel_while_running_kills(self) -> None:
        self.assertTrue(self._cancel_midway(hang=True).instances[0].killed)

    def test_cancel_while_uploading_kills(self) -> None:
        cls = self._cancel_midway(hang_upload=True)
        self.assertEqual(cls.instances[0].written, [])
        self.assertTrue(cls.instances[0].killed)

    def test_cancel_while_creating_propagates(self) -> None:
        # 생성 응답을 받기 전이라 지울 핸들이 없다 — 샌드박스 수명(SANDBOX_LIFETIME_SECONDS)이 정리를 맡는다
        cls = self._cancel_midway(hang_create=True)
        self.assertTrue(cls.create_started)
        self.assertEqual(cls.instances, [])

    def test_slots_full_is_unavailable(self) -> None:
        cls, p = _scenario()

        async def go() -> dict:
            sandbox_service._slots = asyncio.Semaphore(0)
            return await sandbox_service.run_exec_check(IMPL, TEST)

        with p, patch.object(sandbox_service, "SLOT_WAIT_SECONDS", 0.01):
            r = run(go())
        self.assertEqual((r["status"], r["disable"]), ("unavailable", False))
        self.assertEqual(cls.instances, [])


class FixedTestPlacementTest(unittest.TestCase):
    def test_fixed_test_replaces_in_place_and_keeps_main_guard_order(self) -> None:
        from agents.exec_check import with_fixed_test

        code = IMPL + "\ndef smoke_test():\n    return {}\n\nif __name__ == '__main__':\n    smoke_test()\n"
        out = with_fixed_test(code, TEST)
        self.assertLess(out.index("def smoke_test"), out.index("if __name__"))
        self.assertNotIn("return {}", out)
        self.assertIn(TEST, out)
        self.assertEqual(with_fixed_test(IMPL, TEST).rstrip().split("\n")[-1], TEST.rstrip().split("\n")[-1])
        self.assertEqual(with_fixed_test(CODE, None), CODE)

    def test_missing_test_goes_before_main_guard(self) -> None:
        from agents.exec_check import with_fixed_test

        code = IMPL + "\nif __name__ == '__main__':\n    smoke_test()\n"
        out = with_fixed_test(code, TEST)
        self.assertLess(out.index("def smoke_test"), out.index("if __name__"))
        compile(out, "x.py", "exec")


class ExecutorNodeTest(unittest.TestCase):
    def _run(self, state: dict, result: dict) -> tuple[dict, AsyncMock]:
        mock = AsyncMock(return_value=result)
        with patch.object(sandbox_service, "run_exec_check", mock):
            return run(executor.executor_node(state)), mock

    def test_first_test_is_fixed(self) -> None:
        rewritten = IMPL + "\ndef smoke_test():\n    return {}\n"
        out, mock = self._run({"generated_code": rewritten, "smoke_test": TEST, "iteration_count": 2}, {"status": "passed"})
        self.assertEqual(mock.call_args.args[1], TEST)
        self.assertIn(TEST.strip(), out["generated_code"])
        self.assertNotIn("return {}", out["generated_code"])
        self.assertEqual(out["exec_feedback"], "")

    def test_code_issue_feedback_keeps_previous_review(self) -> None:
        out, _ = self._run({"generated_code": CODE, "iteration_count": 1, "review_feedback": "식 (3)이 빠졌다"},
                           {"status": "code_error", "error": "RuntimeError: shape",
                            "runner": {"stage": "run_test", "problems": [], "test_problems": []}})
        self.assertEqual(out["exec_status"], "code_error")
        self.assertIn("RuntimeError: shape", out["exec_feedback"])
        self.assertIn("식 (3)이 빠졌다", out["exec_feedback"])
        self.assertIn("고정된 smoke_test", out["exec_feedback"])

    def test_unavailable_disables_and_later_rounds_skip(self) -> None:
        out, _ = self._run({"generated_code": CODE, "iteration_count": 1}, {"status": "unavailable", "error": "키", "disable": True})
        self.assertTrue(out["exec_disabled"])
        out2, mock = self._run({"generated_code": CODE, "iteration_count": 2, "exec_disabled": True}, {"status": "passed"})
        mock.assert_not_called()
        self.assertEqual(out2["exec_status"], "unavailable")


class RoutingTest(unittest.TestCase):
    def test_after_executor(self) -> None:
        for status in ("code_error", "timeout", "oom"):
            self.assertEqual(router.after_executor({"exec_status": status, "iteration_count": 1}), "coder")
            self.assertEqual(router.after_executor({"exec_status": status, "iteration_count": 3}), END)
        for status in ("passed", "test_error", "unavailable"):
            self.assertEqual(router.after_executor({"exec_status": status, "iteration_count": 3}), "reviewer")

    def test_llm_errors_end(self) -> None:
        self.assertEqual(router.after_coder({"error": "코드 생성 실패"}), END)
        self.assertEqual(router.route({"error": "리뷰 응답 형식 오류", "iteration_count": 1}), END)
        self.assertEqual(router.route({"review_passed": False, "iteration_count": 1}), "coder")


class FinalResultTest(unittest.TestCase):
    def test_last_round_exec_failure_drops_earlier_review(self) -> None:
        # 이전 회차의 판정이 남아 있어도 마지막 코드가 실행에 실패해 검토를 거치지 않았으면 통과로 보이지 않는다
        r = agent_service._code_result({"exec_status": "code_error", "review_passed": True, "review_feedback": "좋음",
                                        "generated_code": "x", "iteration_count": 3})
        self.assertEqual((r["review_passed"], r["review_feedback"], r["exec_status"]), (False, "", "code_error"))
        r = agent_service._code_result({"exec_status": "passed", "review_passed": True, "review_feedback": "좋음"})
        self.assertTrue(r["review_passed"])

    def test_llm_error_is_stored_separately_from_verdict(self) -> None:
        rec = agent_service._code_record({"exec_status": "passed", "review_passed": False, "error": "리뷰 응답 형식 오류"})
        self.assertEqual(rec["run_error"], "리뷰 응답 형식 오류")
        self.assertNotIn("error", rec)
        self.assertEqual(agent_service._code_record({"exec_status": "passed"})["run_error"], "")


class GraphLoopTest(unittest.TestCase):
    """실제 분석 그래프를 LLM·E2B 없이 돌린다 — Analyzer·Coder·Reviewer는 가짜, 실행 결과는 순서대로."""

    def _run_graph(self, exec_results: list[dict], review_passed: bool = True) -> tuple[dict, int, int]:
        from agents import graph as graph_module

        calls = {"coder": 0, "reviewer": 0}

        async def coder(state):  # noqa: ANN001, ANN202
            calls["coder"] += 1
            return {"generated_code": CODE, "iteration_count": state.get("iteration_count", 0) + 1, "error": None}

        async def reviewer(state):  # noqa: ANN001, ANN202
            calls["reviewer"] += 1
            return {"review_passed": review_passed, "review_feedback": "fb", "error": None}

        results = iter(exec_results)
        with patch.object(graph_module, "analyzer_node", AsyncMock(return_value={})), \
             patch.object(graph_module, "coder_node", coder), patch.object(graph_module, "reviewer_node", reviewer), \
             patch.object(sandbox_service, "run_exec_check", AsyncMock(side_effect=lambda *a: next(results))):
            g = graph_module.build_analyze_graph()
            state = run(g.ainvoke(agent_service._make_initial_state("analyze", "q", pdf_pages=["p"])))
        return state, calls["coder"], calls["reviewer"]

    def test_three_exec_failures_end_without_review(self) -> None:
        fail = {"status": "code_error", "error": "boom"}
        state, coder_calls, reviewer_calls = self._run_graph([fail, fail, fail])
        self.assertEqual((coder_calls, reviewer_calls), (3, 0))
        self.assertFalse(agent_service._code_result(state)["review_passed"])

    def test_fix_then_review(self) -> None:
        state, coder_calls, reviewer_calls = self._run_graph([{"status": "code_error", "error": "boom"}, {"status": "passed"}])
        self.assertEqual((coder_calls, reviewer_calls), (2, 1))
        self.assertEqual(state["exec_status"], "passed")

    def test_unavailable_goes_to_reviewer(self) -> None:
        state, coder_calls, reviewer_calls = self._run_graph([{"status": "unavailable", "error": "x", "disable": True}])
        self.assertEqual((coder_calls, reviewer_calls, state["exec_status"]), (1, 1, "unavailable"))


@unittest.skipUnless(os.environ.get("RUN_E2B_TESTS") == "1", "실제 E2B 테스트는 RUN_E2B_TESTS=1일 때만 (과금)")
class RealE2BTest(unittest.TestCase):
    def setUp(self) -> None:
        sandbox_service._slots = asyncio.Semaphore(4)

    def test_pass_fail_and_network(self) -> None:
        ok = "def smoke_test():\n    y = lin(torch.ones(2, 4)); y.sum().backward()\n    return {'outputs': [y], 'grad_params': [lin.weight]}\n"
        self.assertEqual(run(sandbox_service.run_exec_check(IMPL, ok))["status"], "passed")
        bad = "def smoke_test():\n    return {'outputs': [lin(torch.ones(2, 5))], 'grad_params': []}\n"
        self.assertEqual(run(sandbox_service.run_exec_check(IMPL, bad))["status"], "code_error")
        # TCP 연결은 E2B 쪽에서 받아 주지만 응답 없이 끊긴다 (2026-11-07 관측) — HTTP 응답을 받지 못해야 한다
        net = ("def smoke_test():\n    import urllib.request\n    urllib.request.urlopen('http://1.1.1.1', timeout=5)\n"
               "    return {'outputs': [torch.ones(1)], 'grad_params': []}\n")
        self.assertEqual(run(sandbox_service.run_exec_check(IMPL, net))["status"], "code_error")


if __name__ == "__main__":
    unittest.main()
