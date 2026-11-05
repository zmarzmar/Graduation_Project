import asyncio, json, logging, sys
logging.disable(logging.CRITICAL)
from agents.nodes.planner import planner_node
from agents.nodes.researcher import researcher_node

QUERIES = ["attention mechanism", "LoRA fine-tuning", "diffusion model", "mixture of experts",
           "retrieval augmented generation", "어텐션 메커니즘"]

async def main() -> None:
    out = {}
    for q in QUERIES:
        state = {"mode": "search", "user_query": q}
        state.update(await planner_node(state))
        r = await researcher_node(state)
        out[q] = {"keywords": json.loads(state["plan"]).get("search_keywords"),
                  "papers": [(p.get("citation_count"), (p.get("published_at") or "")[:10], bool(p.get("arxiv_id")), p.get("title", "")[:70]) for p in r["papers"]]}
        await asyncio.sleep(3)
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)

asyncio.run(main())
