import json
import logging
from pathlib import Path

from schemas.paper import DailyKeywordsResponse, TrendTopicsBasis

logger = logging.getLogger(__name__)

# 사람이 최근 논문을 근거로 고른 추천 주제 (월 1회 갱신 — 방법은 CLAUDE.md 2-4절)
_TOPICS_FILE = Path(__file__).resolve().parent.parent / "data" / "trend_topics.json"

# 파일을 못 읽을 때의 일반 검색 예시 — 최신 추천처럼 보이지 않게 basis 없이 돌려준다
_FALLBACK_KEYWORDS = [
    "large language model",
    "vision transformer",
    "reinforcement learning",
    "multimodal",
    "agent",
    "mixture of experts",
]


def get_trend_topics(path: Path = _TOPICS_FILE) -> DailyKeywordsResponse:
    """트렌드 화면 추천어와 선정 근거. 파일이 없거나 값의 형식이 틀리면 basis=None인 일반 예시."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keywords = [t["keyword"] for t in data["topics"]][:6]
        # 손으로 고치는 파일이라 값까지 검증한다 (null·숫자 추천어, 형식이 틀린 기준 정보)
        if not keywords or not all(isinstance(k, str) and k.strip() for k in keywords):
            raise ValueError("추천어가 비었거나 문자열이 아님")
        basis = TrendTopicsBasis.model_validate(data)
    except Exception as e:
        logger.warning(f"[TrendTopics] 추천 주제 파일을 쓸 수 없음 — 일반 예시 사용: {type(e).__name__}: {e}")
        return DailyKeywordsResponse(keywords=list(_FALLBACK_KEYWORDS), basis=None)
    return DailyKeywordsResponse(keywords=keywords, basis=basis)
