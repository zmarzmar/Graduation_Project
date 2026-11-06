"""구현 대상이 아닌 논문이면 Coder가 코드를 만들지 않고 루프를 끝내는지 (LLM을 부르지 않는다).

실행: cd BE && uv run python -m unittest discover -s tests -v
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langgraph.graph import END

from agents.graph import _after_coder, build_analyze_graph
from agents.nodes import coder
from agents.nodes.coder import _skip_reason, coder_node
from evals.loop_eval import collect_rounds, end_reason

_STATE = {"pdf_text": "논문 본문", "pdf_pages": ["논문 본문"], "papers": [], "key_formulas": []}


def _run_coder(content: str, iteration: int) -> dict:
    with patch.object(coder, "_llm") as llm:
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
        return asyncio.run(coder_node({**_STATE, "iteration_count": iteration}))


class SkipReasonTest(unittest.TestCase):
    def test_marker_on_first_line(self) -> None:
        self.assertEqual(_skip_reason("NOT_IMPLEMENTABLE: 서베이 논문입니다."), "서베이 논문입니다.")
        # 앞의 빈 줄·코드 블록 표시는 건너뛴다
        self.assertEqual(_skip_reason("\n```\nNOT_IMPLEMENTABLE: 이론 논문\n```"), "이론 논문")
        self.assertEqual(_skip_reason("```python\nNOT_IMPLEMENTABLE: 서베이\n```"), "서베이")
        self.assertEqual(_skip_reason("`NOT_IMPLEMENTABLE: 서베이`"), "서베이")
        # 이유가 비어도 생략으로 본다
        self.assertTrue(_skip_reason("NOT_IMPLEMENTABLE:"))

    def test_code_is_not_skip(self) -> None:
        self.assertIsNone(_skip_reason("```python\nimport torch\n```"))
        # 코드 안에 표시 문자열이 나와도 첫 줄이 아니면 생략이 아니다
        self.assertIsNone(_skip_reason("import torch\n# NOT_IMPLEMENTABLE: 아님"))
        self.assertIsNone(_skip_reason(""))


class CoderNodeSkipTest(unittest.TestCase):
    def test_first_round_skip_returns_reason_without_code(self) -> None:
        result = _run_coder("NOT_IMPLEMENTABLE: 벤치마크 공개 논문입니다.", iteration=0)
        self.assertEqual(result["generated_code"], "")
        self.assertEqual(result["code_skip_reason"], "벤치마크 공개 논문입니다.")
        self.assertIsNone(result["error"])

    def test_revision_round_never_skips(self) -> None:
        # 수정 회차에는 생략 선택지를 주지 않는다 — 표시가 와도 코드 회차로 처리한다
        result = _run_coder("NOT_IMPLEMENTABLE: 안 함", iteration=1)
        self.assertNotIn("code_skip_reason", result)

    def test_first_round_prompt_offers_skip_and_revision_does_not(self) -> None:
        for iteration, offered in ((0, True), (1, False)):
            with patch.object(coder, "_llm") as llm:
                llm.ainvoke = AsyncMock(return_value=AIMessage(content="import torch"))
                asyncio.run(coder_node({**_STATE, "iteration_count": iteration}))
                prompt = llm.ainvoke.call_args.args[0][1].content
            self.assertEqual("NOT_IMPLEMENTABLE" in prompt, offered)


class GraphSkipTest(unittest.TestCase):
    def test_after_coder(self) -> None:
        self.assertEqual(_after_coder({"code_skip_reason": "서베이"}), END)
        self.assertEqual(_after_coder({"code_skip_reason": ""}), "reviewer")
        self.assertEqual(_after_coder({}), "reviewer")

    def test_graph_ends_without_reviewer_when_skipped(self) -> None:
        # 실제 그래프로 돌려 Reviewer가 불리지 않는지 본다 (Analyzer·Coder LLM은 가짜)
        async def run() -> list[str]:
            nodes: list[str] = []
            with patch("agents.nodes.analyzer._llm") as a_llm, patch.object(coder, "_llm") as c_llm, \
                    patch("agents.nodes.reviewer._llm") as r_llm:
                a_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"summary": "s"}'))
                c_llm.ainvoke = AsyncMock(return_value=AIMessage(content="NOT_IMPLEMENTABLE: 서베이"))
                r_llm.ainvoke = AsyncMock(side_effect=AssertionError("Reviewer가 불리면 안 된다"))
                graph = build_analyze_graph()
                async for chunk in graph.astream({**_STATE, "mode": "pdf", "iteration_count": 0}, stream_mode="updates"):
                    nodes += list(chunk)
            return nodes

        self.assertEqual(asyncio.run(run()), ["analyzer", "coder"])


class LoopEvalSkipTest(unittest.TestCase):
    def test_end_reason_not_implementable(self) -> None:
        rounds = collect_rounds([("coder", {"generated_code": "", "code_skip_reason": "서베이", "iteration_count": 1})])
        self.assertEqual(rounds[0]["skip_reason"], "서베이")
        self.assertEqual(end_reason(rounds, None), "not_implementable")


if __name__ == "__main__":
    unittest.main()
