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
    # 삭제 대기 중인 문서의 벡터 색인 정리를 다시 시도한다 — 지난 실행에서 Chroma 삭제가 실패했거나
    # 유예 시간이 지나지 않아 남은 툼스톤을 여기서 마무리한다. 실패해도 서버 기동은 막지 않는다.
    try:
        await rag_service.purge_pending_documents()
    except Exception as e:
        logger.error(f"기동 시 문서 정리 실패 (무시): {e}")
    yield


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
