"""papers.json에서 주제 후보별 논문을 찾는다 (제목·초록·HF 키워드에 표현이 있는 논문)."""
import json, re, sys

TOPICS = {
    "RL for LLM reasoning (RLVR · GRPO)": r"\bgrpo\b|group relative policy optimi|verifiable reward|\brlvr\b|reinforcement learning[^.]{0,60}(reasoning|llm|language model)",
    "LLM agents / coding agents": r"\b(llm|coding|software|web|gui|computer[- ]use|deep research) agents?\b|agentic",
    "On-policy distillation / self-distillation": r"on-policy (self-)?distillation|self-distillation",
    "Vision-language-action models (robotics)": r"vision-language-action|\bvla\b",
    "Multimodal LLMs / vision-language models": r"multimodal large language model|\bmllms?\b|vision-language models?\b|\bvlms?\b",
    "Mixture-of-experts": r"mixture[- ]of[- ]experts|\bmoe\b",
    "Speculative decoding / efficient inference": r"speculative decoding|kv[- ]cache|efficient inference",
    "Flow matching / diffusion generation": r"flow matching|diffusion (model|transformer)s?",
    "Recursive self-improvement": r"self-improv",
    "World models": r"world models?\b",
    "Test-time scaling": r"test-time (scaling|compute)",
}

papers = json.load(open("papers.json"))["papers"]
def text(p): return " ".join([p["title"], p["summary"], " ".join(p["ai_keywords"])]).lower()
out = {}
for name, pat in TOPICS.items():
    rx = re.compile(pat)
    hits = [p for p in papers if rx.search(text(p))]
    title_hits = [p for p in hits if rx.search(p["title"].lower())]
    top = sorted(title_hits or hits, key=lambda p: -(p["upvotes"] or 0))[:4]
    out[name] = {"pattern": pat, "matched": len(hits), "title_matched": len(title_hits),
                 "examples": [{"id": p["id"], "title": p["title"], "upvotes": p["upvotes"], "date": p["daily_date"]} for p in top]}
json.dump({"total_papers": len(papers), "topics": out}, open("candidates.json", "w"), ensure_ascii=False, indent=1)
for name, v in sorted(out.items(), key=lambda kv: -kv[1]["matched"]):
    print(f"\n## {name} — 일치 {v['matched']}편 (제목 {v['title_matched']}편)")
    for e in v["examples"]: print(f"   [{e['upvotes']:>3}] {e['id']} {e['title'][:85]}")
