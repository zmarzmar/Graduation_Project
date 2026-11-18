"""조사용: before/after 답변의 claim을 '인용이 문장을 뒷받침하는가'로 판정 (gpt-5.4-mini, 판정기 프롬프트는 evals/qa_eval.py 그대로)."""
import asyncio, json, sys
from langchain_openai import ChatOpenAI
from core.config import settings
from evals import qa_eval
S = "../eval-results/2026-11-qa-richer/"
qa_eval._judge = ChatOpenAI(model="gpt-5.4-mini", reasoning_effort="medium", api_key=settings.openai_api_key,
                            timeout=120, max_retries=1).with_structured_output(qa_eval.Verdict)

async def main() -> None:
    for label in sys.argv[1:]:
        rows = json.load(open(S + f"qa_{label}_claims.json"))
        vs = await asyncio.gather(*(qa_eval.judge_support(r["q"], r["text"], r["quotes"]) for r in rows))
        # 문장별 판정을 저장한다 — NOTES의 비율을 원자료로 대조할 수 있게 (Codex 리뷰 지적)
        json.dump([{**r, "supported": v.supported, "reason": v.reason} for r, v in zip(rows, vs)],
                  open(S + f"qa_{label}_verdicts.json", "w"), ensure_ascii=False, indent=1)
        bad = [(r, v) for r, v in zip(rows, vs) if not v.supported]
        print(f"[{label}] 문장 {len(rows)}개 중 인용이 뒷받침 안 함 {len(bad)}개")
        for r, v in bad[:4]:
            print("   -", r["text"][:120], "|", v.reason[:160])

asyncio.run(main())
