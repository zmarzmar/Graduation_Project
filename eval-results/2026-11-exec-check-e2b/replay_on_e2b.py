"""#60 실험(로컬 Docker)의 회차별 코드·고정 smoke_test를 그대로 E2B에서 다시 실행해 상태를 비교한다 (환경 차이 확인).

실행 (BE에서): uv run python ../eval-results/2026-11-exec-check-e2b/replay_on_e2b.py
같은 (코드, 테스트) 쌍은 한 번만 돌린다. 결과: 같은 폴더의 replay.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from services import sandbox_service  # noqa: E402

SRC = Path("../eval-results/2026-11-exec-check")
OUT = Path("../eval-results/2026-11-exec-check-e2b/replay.json")


def collect() -> list[dict]:
    seen: dict[tuple[str, str], dict] = {}
    for f in ("pilot_explicit_dropout.json", "main_9.json", "rerun_3.json"):
        for r in json.loads((SRC / f).read_text())["results"]:
            for arm, a in r.get("arms", {}).items():
                for x in a.get("rounds", []):
                    key = (x["code"], x["smoke_test"] or "")
                    item = seen.setdefault(key, {"file": f, "paper": r["paper"], "where": [], "docker": x["exec"]["status"]})
                    item["where"].append(f"{arm}#{x['iteration']}")
                    item.setdefault("docker_all", set()).add(x["exec"]["status"])
                    item["code"], item["smoke_test"] = key
    return list(seen.values())


async def main() -> None:
    items = collect()
    print(f"고유 (코드, 테스트) {len(items)}개", flush=True)

    queue = asyncio.Semaphore(4)  # 운영의 동시 상한(4)을 넘기면 대기 10초 뒤 건너뛰므로, 여기서 줄을 세운다

    async def one(it: dict) -> None:
        async with queue:
            t0 = time.perf_counter()
            r = await sandbox_service.run_exec_check(it["code"], it["smoke_test"] or None)
        it.update(e2b=r["status"], e2b_error=(r.get("error") or "")[-300:], seconds=round(time.perf_counter() - t0, 1))
        print(f"{it['file']:28} {it['paper']:28} {','.join(it['where']):22} docker={it['docker']:11} e2b={it['e2b']:11} {it['seconds']}s", flush=True)

    await asyncio.gather(*(one(it) for it in items))
    for it in items:
        it["docker_all"] = sorted(it["docker_all"])
        del it["code"], it["smoke_test"]
    same = sum(it["e2b"] == it["docker"] for it in items)
    secs = sorted(it["seconds"] for it in items)
    summary = {"pairs": len(items), "same_status": same, "different": [it for it in items if it["e2b"] != it["docker"]],
               "seconds_min_median_max": [secs[0], secs[len(secs) // 2], secs[-1]]}
    OUT.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:3000])


asyncio.run(main())
