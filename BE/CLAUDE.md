# BE/CLAUDE.md

This file provides backend-specific guidance for the AI-arXiv Analyst project.

---

## Overview

FastAPI 기반 백엔드. LangGraph 에이전트를 실행하고 결과를 SSE로 프론트에 스트리밍. 논문 데이터 수집 및 DB 연동 담당.

---

## Tech Stack

- **Framework**: FastAPI (Python 3.11)
- **Package Manager**: uv
- **AI Core**: LangGraph (agents/ 폴더)
- **Vector DB**: ChromaDB
- **Relational DB**: PostgreSQL (SQLAlchemy)
- **HTTP Client**: httpx (비동기)

---

## Folder Structure

```
BE/
├── main.py                         # FastAPI 앱 진입점, 라우터 등록
├── pyproject.toml                  # uv 의존성 관리
├── .env                            # 환경변수
├── core/
│   ├── config.py                   # 환경변수 로드 (pydantic-settings)
│   └── dependencies.py             # 공통 의존성 (DB 세션 등)
├── routers/
│   ├── paper.py                    # 논문 검색/조회 API
│   └── agent.py                    # 에이전트 실행 API (SSE 스트리밍)
├── services/
│   ├── arxiv_service.py            # arXiv API 연동
│   ├── semantic_scholar_service.py # Semantic Scholar API 연동
│   ├── agent_service.py            # LangGraph 에이전트 실행
│   └── vector_service.py           # ChromaDB 벡터 저장/검색
├── agents/
│   ├── graph.py                    # LangGraph 전체 그래프 정의
│   ├── state.py                    # 에이전트 공유 상태 정의
│   └── nodes/
│       ├── planner.py
│       ├── researcher.py
│       ├── coder.py
│       ├── reviewer.py
│       └── router.py
└── models/
    └── paper.py                    # Pydantic 모델 (DB 붙이면 schemas/로 분리)
```

---

## Dev Commands

```bash
uv sync                                  # 의존성 설치 (pull 후 항상 실행)
uv run uvicorn main:app --reload         # 개발 서버 실행 (localhost:8000)
uv add <패키지명>                         # 패키지 추가
```

---

## API Endpoints

```
GET  /health                        # 서버 상태 확인
GET  /api/v1/papers/search          # 논문 검색 (arXiv + Semantic Scholar)
POST /api/v1/agent/pdf              # PDF 업로드 모드 (SSE 스트리밍)
POST /api/v1/agent/search           # 키워드 검색 모드 (SSE 스트리밍)
POST /api/v1/agent/trend            # 트렌드 브리핑 모드 (SSE 스트리밍)
```

모든 라우터는 `/api/v1` prefix 사용.

---

## Agent Pipeline

```
Planner → Researcher → Coder → Reviewer → Router
                                    ↑           |
                                    |___ 낙제 __|
```

- **Planner**: gpt-4o-mini 사용 (간단한 계획 수립)
- **Researcher**: gpt-4o-mini 사용 (논문 검색 및 수집)
- **Coder**: o4-mini 사용 (코드 생성, 추론 능력 중요)
- **Reviewer**: o4-mini 사용 (코드 검증, 추론 능력 중요)
- **Router**: 조건부 분기 (통과/낙제 판단)

---

## SSE 스트리밍 방식

에이전트 각 노드 실행 시 진행 상황을 실시간으로 프론트에 전송.

```python
# routers/agent.py
from fastapi.responses import StreamingResponse

@router.post("/agent/search")
async def run_search_agent(request: SearchRequest):
    return StreamingResponse(
        agent_service.stream_agent(request),
        media_type="text/event-stream"
    )
```

---

## Coding Conventions

- **타입 힌트**: 모든 함수에 필수
- **비동기**: DB 조회, API 호출은 항상 async/await 사용
- **비즈니스 로직**: 반드시 services/에만 작성. routers/에서 직접 작성 금지
- **에이전트 로직**: 반드시 agents/에만 작성. services/에서 직접 작성 금지
- **변수명**: snake_case
- **주석**: 한국어로 작성

---

## DB 관련 (추후 추가 예정)

DB 붙일 때 아래 구조로 확장:
```
models/   → SQLAlchemy @Entity (DB 테이블)
schemas/  → Pydantic DTO (현재 models/에 있는 것 이동)
crud/     → DB 쿼리 함수 (@Repository 역할)
alembic/  → DB 마이그레이션 (Flyway 역할)
```

---

## Environment Variables

```
OPENAI_API_KEY=
DATABASE_URL=
CHROMA_HOST=localhost
CHROMA_PORT=8001
HUGGINGFACE_TOKEN=
```

---

## Important Rules

- CORS 설정 필수 — 없으면 FE(localhost:3000)에서 호출 불가
- 비즈니스 로직은 services/에서만
- 에이전트 노드는 agents/nodes/에서만
- RAG/ 폴더는 레거시 — 참고만 하고 BE/agents/에 새로 구현할 것
- API 키는 절대 하드코딩 금지
