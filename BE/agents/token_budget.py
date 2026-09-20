"""모델 호출의 입력·출력 토큰 예산을 한 곳에서 관리한다."""

from functools import lru_cache

import tiktoken
from langchain_core.messages import BaseMessage

# 컨텍스트 한도 (OpenAI 공식 문서 기준)
CONTEXT_WINDOW = {"gpt-4o-mini": 128_000, "o4-mini": 200_000}

# 호출당 출력 예산 — ChatOpenAI(max_tokens=...)로 API에 강제한다.
# o4-mini는 보이는 답변 외에 추론 토큰도 이 예산에서 쓰므로 관측된 출력 길이(1.4k~5.6k)보다 넉넉히 잡는다.
REASONING_OUTPUT_BUDGET = 32_000
ANALYZER_OUTPUT_BUDGET = 16_384  # gpt-4o-mini 최대 출력

# 채팅 형식 오버헤드 (메시지당 구분 토큰 + 응답 시작 토큰)
_PER_MESSAGE_OVERHEAD = 4
_REPLY_PRIMER = 3


class TokenizerUnavailableError(RuntimeError):
    """토크나이저를 쓸 수 없어 길이를 검증할 수 없다. 부정확한 추정으로 계속 진행하지 않는다."""


class TokenBudgetError(ValueError):
    """입력 + 출력 예산이 모델 컨텍스트 한도를 넘는다."""


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # gpt-4o-mini·o4-mini 공용 인코딩. Docker 이미지에는 빌드 시 미리 받아둔다 (TIKTOKEN_CACHE_DIR).
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 실제 토크나이저로 센다."""
    try:
        encoding = _encoding()
    except Exception as e:
        raise TokenizerUnavailableError(
            "토크나이저를 불러오지 못해 논문 길이를 검증할 수 없습니다. 잠시 후 다시 시도해주세요."
        ) from e
    return len(encoding.encode(text, disallowed_special=()))


def ensure_request_fits(model: str, messages: list[BaseMessage], output_budget: int) -> int:
    """호출 직전에 '전체 입력 + 출력 예산'이 컨텍스트 한도 안인지 확인하고 입력 토큰 수를 반환한다.

    본문 한도(max_paper_tokens)는 본문만 제한한다 — 이전 코드·리뷰 피드백 등 회차마다 달라지는 입력은 여기서 잡는다.
    """
    input_tokens = _REPLY_PRIMER + sum(_PER_MESSAGE_OVERHEAD + count_tokens(str(m.content)) for m in messages)
    limit = CONTEXT_WINDOW[model]
    if input_tokens + output_budget > limit:
        raise TokenBudgetError(
            f"입력이 너무 깁니다 — 입력 {input_tokens:,} + 출력 예산 {output_budget:,} 토큰이 "
            f"{model} 한도 {limit:,} 토큰을 넘습니다."
        )
    return input_tokens


def ensure_complete(response: BaseMessage, output_budget: int) -> None:
    """출력 예산 안에 응답을 끝내지 못했으면 잘린 결과를 쓰지 않고 오류로 처리한다."""
    metadata = getattr(response, "response_metadata", None) or {}
    if metadata.get("finish_reason") == "length":
        raise TokenBudgetError(f"모델이 출력 예산 {output_budget:,} 토큰 안에 응답을 끝내지 못했습니다.")
