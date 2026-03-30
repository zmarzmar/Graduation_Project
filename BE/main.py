import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings  # noqa: F401
from routers import agent, paper, mypage

# 터미널 로그 포맷 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s \033[90m|\033[0m %(levelname)-8s \033[90m|\033[0m %(message)s",
    datefmt="%H:%M:%S",
)
# uvicorn 기본 로거는 그대로 유지
logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.access").propagate = False

app = FastAPI(
    title="AI-arXiv Analyst API",
    version="0.1.0",
)

# CORS 설정 — FE(localhost:3000)에서 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paper.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(mypage.router, prefix="/api/v1")


@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "version": "0.1.0"}
