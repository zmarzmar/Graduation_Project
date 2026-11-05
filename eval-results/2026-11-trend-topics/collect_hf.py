"""HF Daily Papers를 날짜별로 모아 papers.json으로 저장한다 (추천 주제 선정의 근거 자료).

실행: python3 collect_hf.py 2026-10-08 2026-11-04 papers.json
"""
import json
import sys
import time
import urllib.request
from datetime import date, timedelta

start, end, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
papers: dict[str, dict] = {}
per_day: dict[str, int] = {}
d = start
while d <= end:
    url = f"https://huggingface.co/api/daily_papers?date={d.isoformat()}&limit=100"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            items = json.load(r)
    except Exception as e:  # 실패한 날은 0이 아니라 실패로 기록한다
        per_day[d.isoformat()] = -1
        print(d, "failed", type(e).__name__, file=sys.stderr)
    else:
        per_day[d.isoformat()] = len(items)
        for it in items:
            p = it.get("paper") or {}
            if not p.get("id"):
                continue
            papers.setdefault(p["id"], {
                "id": p["id"],
                "daily_date": d.isoformat(),
                "title": p.get("title", ""),
                "summary": p.get("summary", ""),
                "upvotes": p.get("upvotes"),
                "ai_keywords": p.get("ai_keywords") or [],
                "url": f"https://huggingface.co/papers/{p['id']}",
            })
    d += timedelta(days=1)
    time.sleep(1)

json.dump({"source": "https://huggingface.co/api/daily_papers?date=…", "range": [sys.argv[1], sys.argv[2]],
           "collected_at": date.today().isoformat(), "per_day": per_day, "papers": list(papers.values())},
          open(out, "w"), ensure_ascii=False, indent=1)
print("papers", len(papers), "days", len(per_day), "failed_days", sum(1 for v in per_day.values() if v < 0))
