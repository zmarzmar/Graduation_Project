"""노드 실행 중 실시간 로그를 SSE 스트림으로 전송하는 유틸리티.

사용법:
    from agents.log_stream import emit_log

    async def some_node(state):
        emit_log("researcher", "Semantic Scholar 검색 중...")
        # ... 작업 ...
        emit_log("researcher", "논문 5편 수집 완료")
"""

import asyncio
import contextvars

# 각 요청마다 독립적인 Queue를 가리키는 컨텍스트 변수
_log_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "_log_queue", default=None
)


def set_log_queue(q: asyncio.Queue) -> None:
    """stream_agent에서 요청 시작 시 호출해 Queue를 등록한다."""
    _log_queue.set(q)


def emit_log(node: str, message: str) -> None:
    """노드 내부에서 호출해 실시간 로그를 SSE 스트림으로 전송한다.
    Queue가 없으면 무시 (테스트/직접 실행 시 안전).
    """
    q = _log_queue.get()
    if q is not None:
        try:
            q.put_nowait(("log", node, message))
        except asyncio.QueueFull:
            pass
