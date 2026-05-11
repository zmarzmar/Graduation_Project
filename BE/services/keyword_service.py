import json
import logging
from datetime import date

from openai import AsyncOpenAI

from core.config import settings

logger = logging.getLogger(__name__)

# 당일 캐시 — 날짜가 바뀌면 자동으로 새로 fetch
_cache: dict[date, list[str]] = {}

_FALLBACK_KEYWORDS = [
    "large language model",
    "vision transformer",
    "reinforcement learning",
    "multimodal",
    "agent",
]

_PROMPT = (
    "Today is {date}. "
    "List exactly 5 trending AI/ML research keywords that are hot right now. "
    "Return a JSON array of strings only, no explanation. "
    "Each keyword must be 2–4 words in English. "
    'Example: ["mixture of experts", "chain of thought", "diffusion model", "state space model", "RLHF"]'
)


async def get_daily_keywords() -> list[str]:
    """오늘 날짜 기준 트렌드 키워드 5개 반환 (당일 캐시)"""
    today = date.today()
    if today in _cache:
        return _cache[today]

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _PROMPT.format(date=today.isoformat())}],
            temperature=0.7,
            max_tokens=100,
        )
        raw = response.choices[0].message.content or "[]"
        # ```json ... ``` 블록 제거
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        keywords: list[str] = json.loads(raw)
        if not isinstance(keywords, list) or len(keywords) == 0:
            raise ValueError("빈 응답")
        keywords = [str(k) for k in keywords[:5]]
        _cache[today] = keywords
        logger.info(f"[DailyKeywords] 오늘({today}) 키워드 갱신: {keywords}")
        return keywords
    except Exception as e:
        logger.warning(f"[DailyKeywords] OpenAI 호출 실패, 폴백 사용: {e}")
        return _FALLBACK_KEYWORDS
