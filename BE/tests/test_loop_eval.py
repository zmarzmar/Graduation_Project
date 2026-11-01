"""자기수정 루프 평가의 정적 검사·회차 모으기 테스트 (LLM을 부르지 않는다).

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import builtins
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx
from langgraph.graph import END

from agents.nodes.router import route
from evals import loop_eval
from evals.loop_eval import MAX_ITERATIONS, collect_rounds, end_reason
from evals.static_check import static_check


class StaticCheckTest(unittest.TestCase):
    def test_reports_syntax_error_with_line(self) -> None:
        result = static_check("import torch\n\ndef forward(x:\n    return x\n")
        self.assertFalse(result["syntax_ok"])
        self.assertTrue(result["syntax_error"].startswith("3행"))
        self.assertEqual(result["imports"], [])

    def test_empty_code_is_not_a_pass(self) -> None:
        for code in ("", "  \n"):
            result = static_check(code)
            self.assertFalse(result["syntax_ok"])
            self.assertEqual(result["syntax_error"], "코드 없음")

    def test_null_byte_is_a_syntax_failure(self) -> None:
        self.assertFalse(static_check("x = 1\x00")["syntax_ok"])

    def test_lists_imports_and_marks_unknown_ones(self) -> None:
        code = (
            "import math, torch.nn as nn\n"
            "from torch.nn import functional as F\n"
            "from einops import rearrange\n"
            "def f():\n"
            "    import transformers.models\n"
            "    from .utils import helper\n"
            "    from . import sibling\n"
        )
        result = static_check(code)
        self.assertTrue(result["syntax_ok"])
        self.assertEqual(result["imports"], [".", ".utils", "einops", "math", "torch", "transformers"])
        self.assertEqual(result["unverified_imports"], [".", ".utils", "einops", "transformers"])

    def test_does_not_execute_the_code(self) -> None:
        marker = "_loop_eval_executed"
        self.addCleanup(lambda: vars(builtins).pop(marker, None))
        result = static_check(f"import builtins\nbuiltins.{marker} = True\nraise SystemExit('executed')\n")
        self.assertTrue(result["syntax_ok"])
        self.assertFalse(hasattr(builtins, marker))


class CollectRoundsTest(unittest.TestCase):
    def test_keeps_every_round_not_only_the_last(self) -> None:
        events = [
            ("analyzer", {"paper_summary": "요약"}),
            ("coder", {"generated_code": "import torch", "iteration_count": 1, "error": None}),
            ("reviewer", {"review_passed": False, "review_feedback": "스케일링 누락"}),
            ("coder", {"generated_code": "import torch\nimport einops", "iteration_count": 2, "error": None}),
            ("reviewer", {"review_passed": True, "review_feedback": "통과"}),
        ]
        rounds = collect_rounds(events)
        self.assertEqual([r["iteration"] for r in rounds], [1, 2])
        self.assertEqual([r["code"] for r in rounds], ["import torch", "import torch\nimport einops"])
        self.assertEqual([r["review_feedback"] for r in rounds], ["스케일링 누락", "통과"])
        self.assertEqual([r["review_passed"] for r in rounds], [False, True])
        self.assertEqual(rounds[1]["static"]["unverified_imports"], ["einops"])
        self.assertEqual(end_reason(rounds, None), "passed")

    def test_round_where_coder_failed_is_recorded_as_such(self) -> None:
        # Coder가 예외로 빈 코드를 내면 Reviewer는 '검토할 코드가 없습니다'로 낙제시킨다
        events = [
            ("coder", {"generated_code": "", "iteration_count": 1, "error": "코드 생성 실패: timeout"}),
            ("reviewer", {"review_passed": False, "review_feedback": "검토할 코드가 없습니다.", "error": "generated_code가 비어 있습니다."}),
        ]
        (only,) = collect_rounds(events)
        self.assertEqual(only["coder_error"], "코드 생성 실패: timeout")
        self.assertEqual(only["reviewer_error"], "generated_code가 비어 있습니다.")
        self.assertFalse(only["static"]["syntax_ok"])

    def test_end_reasons(self) -> None:
        def failed_rounds(count: int) -> list[dict]:
            events = []
            for i in range(1, count + 1):
                events += [("coder", {"generated_code": "x = 1", "iteration_count": i}), ("reviewer", {"review_passed": False})]
            return collect_rounds(events)

        self.assertEqual(end_reason(failed_rounds(MAX_ITERATIONS), None), "max_iterations")
        self.assertEqual(end_reason(failed_rounds(1), None), "incomplete")
        self.assertEqual(end_reason(failed_rounds(1), "boom"), "error")
        self.assertEqual(end_reason([], None), "incomplete")
        # 리뷰가 붙지 않은 채 끊긴 마지막 회차는 상한 도달이 아니다
        unreviewed = collect_rounds([("coder", {"generated_code": "x = 1", "iteration_count": 3})] * MAX_ITERATIONS)
        self.assertEqual(end_reason(unreviewed, None), "incomplete")

    def test_max_iterations_matches_the_router(self) -> None:
        self.assertEqual(route({"review_passed": False, "iteration_count": MAX_ITERATIONS}), END)
        self.assertEqual(route({"review_passed": False, "iteration_count": MAX_ITERATIONS - 1}), "coder")


class _FakeGraph:
    """준비된 노드 갱신분을 흘려보낸다."""

    def __init__(self, events: list[tuple[str, dict]]):
        self.events = events

    async def astream(self, _state: dict, stream_mode: str = "updates"):
        for node, updates in self.events:
            yield {node: updates}


_PASSING_RUN = [
    ("analyzer", {"paper_summary": "요약", "key_formulas": []}),
    ("coder", {"generated_code": "import torch", "iteration_count": 1, "error": None}),
    ("reviewer", {"review_passed": True, "review_feedback": "통과"}),
]


class RunTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_download_keeps_finished_results(self) -> None:
        # 두 번째 논문의 다운로드가 실패해도 첫 번째 논문의 결과는 파일에 남아야 한다
        download = AsyncMock(side_effect=[["page"], httpx.ConnectTimeout("timed out")])
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "results.json")
            with (
                patch.object(loop_eval, "download_pdf_pages", download),
                patch.object(loop_eval, "analyze_graph", _FakeGraph(_PASSING_RUN)),
                redirect_stdout(io.StringIO()),
            ):
                await loop_eval.run(["attention", "lora"], out)
            with open(out, encoding="utf-8") as f:
                first, second = json.load(f)["results"]
        self.assertEqual((first["paper"], first["end_reason"]), ("attention", "passed"))
        self.assertEqual(first["rounds"][0]["code"], "import torch")
        self.assertEqual((second["paper"], second["end_reason"], second["rounds"]), ("lora", "error", []))
        self.assertIn("ConnectTimeout", second["run_error"])

    async def test_analyzer_failure_is_kept_in_the_result(self) -> None:
        # Analyzer는 실패를 예외가 아니라 error 갱신분으로 돌려주고 그래프는 Coder로 계속 간다
        events = [("analyzer", {"paper_summary": "", "key_formulas": [], "error": "LLM API 호출에 실패했습니다."}), *_PASSING_RUN[1:]]
        with (
            patch.object(loop_eval, "download_pdf_pages", AsyncMock(return_value=["page"])),
            patch.object(loop_eval, "analyze_graph", _FakeGraph(events)),
            redirect_stdout(io.StringIO()),
        ):
            result = await loop_eval.run_paper("attention")
        self.assertEqual(result["end_reason"], "passed")
        self.assertEqual(result["analyzer_error"], "LLM API 호출에 실패했습니다.")


if __name__ == "__main__":
    unittest.main()
