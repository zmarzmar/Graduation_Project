"""확정한 추천어로 트렌드 모드 Planner+Researcher를 돌려 나오는 논문을 기록한다 (실제 API)."""
import asyncio, json, logging, sys
logging.disable(logging.CRITICAL)
from agents.nodes.planner import planner_node
from agents.nodes.researcher import researcher_node

TOPICS = ["LLM reinforcement learning", "coding agents", "on-policy distillation",
          "vision-language-action", "world models", "video generation"]

async def main() -> None:
    out = {}
    for q in TOPICS:
        state = {"mode": "trend", "user_query": q}
        state.update(await planner_node(state))
        r = await researcher_node(state)
        out[q] = {"keywords": json.loads(state["plan"]).get("search_keywords"),
                  "papers": [p.get("title", "") for p in r["papers"]]}
        await asyncio.sleep(2)
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    for q, v in out.items():
        print(f"\n## {q}  키워드={v['keywords']}  {len(v['papers'])}편")
        for t in v["papers"]: print("   -", t[:90])

asyncio.run(main())
