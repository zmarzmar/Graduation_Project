import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


@asynccontextmanager
async def log_elapsed(
    logger: logging.Logger,
    event: str,
    **fields: object,
) -> AsyncIterator[None]:
    """서버 전용 성능 로그를 남긴다. emit_log와 달리 SSE로 전송하지 않는다."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(f"[Perf] event={event} elapsed_ms={elapsed_ms} {details}".strip())
