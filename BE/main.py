import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings  # noqa: F401
from routers import admin, agent, auth, mypage, paper, qa
from services import rag_service

# 터미널 로그 포맷 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s \033[90m|\033[0m %(levelname)-8s \033[90m|\033[0m %(message)s",
    datefmt="%H:%M:%S",
)
# uvicorn 기본 로거는 그대로 유지
logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.access").propagate = False

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 삭제된 문서의 벡터 색인 정리를 백그라운드에서 되풀이한다 — 지난 실행에서 남은 툼스톤과
    # 정리 뒤에 늦게 도착한 청크를 마무리한다. 기동을 기다리게 하지 않고, 실패해도 서버는 뜬다.
    purge_task = asyncio.create_task(rag_service.purge_periodically())
    yield
    purge_task.cancel()


app = FastAPI(
    title="AI-arXiv Analyst API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정 — FE(localhost:3000)에서 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(paper.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(mypage.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")


@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "version": "0.1.0"}
