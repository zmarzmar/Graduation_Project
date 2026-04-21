# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

**AI Research Analyst** is a graduation project that builds a Deep Research Agent for analyzing the latest AI research papers and trends.

The system goes beyond simple summarization — it retrieves papers from multiple sources (arXiv, Semantic Scholar, Hugging Face), generates PyTorch/TensorFlow code skeletons based on the paper's Methods section, and runs a **self-correction loop** powered by LangGraph to verify and fix the generated code against the paper's theory.

---

## 3 Core Modes

1. **PDF Upload Mode** — 사용자가 논문 PDF를 업로드하면 자동 분석 후 코드 재현
2. **Keyword Search Mode** — 키워드 입력 시 arXiv + Semantic Scholar에서 논문 검색 후 분석 및 코드 재현
3. **Trend Briefing Mode** — Hugging Face + Semantic Scholar 기반 최신 트렌드 논문 Top N 요약 리포트 생성

---

## Tech Stack

- **Frontend**: Next.js (TypeScript), Tailwind CSS, Zustand, React Query
- **Backend**: FastAPI (Python 3.11)
- **AI Core**: LangGraph, OpenAI API
  - Planner, Researcher: `gpt-4o-mini` (비용 절약)
  - Coder, Reviewer: `o4-mini` (코드 추론 특화)
- **Vector DB**: ChromaDB
- **Relational DB**: PostgreSQL
- **Package Manager**: uv (Python), npm (Node.js)

---

## Project Structure

```
Graduation_Project/
├── CLAUDE.md
├── .claude/
│   └── settings.local.json
├── .env                            # Root environment variables
├── FE/                             # Next.js frontend
│   ├── CLAUDE.md
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── store/
│   └── lib/
│       └── api.ts                  # API 호출 전용 파일
├── BE/                             # FastAPI backend
│   ├── CLAUDE.md
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── dependencies.py
│   ├── routers/
│   │   ├── paper.py
│   │   └── agent.py
│   ├── services/
│   │   ├── arxiv_service.py
│   │   ├── semantic_scholar_service.py
│   │   ├── agent_service.py
│   │   └── vector_service.py
│   ├── agents/                     # LangGraph 에이전트 (핵심)
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── researcher.py
│   │       ├── coder.py
│   │       ├── reviewer.py
│   │       └── router.py
│   └── models/
│       └── paper.py
```

---

## Agent Architecture

LangGraph 기반 멀티 에이전트 파이프라인:

```
Planner → Researcher → Coder → Reviewer → Router
                                    ↑           |
                                    |___ 낙제 __|
```

1. **Planner** — 사용자 입력 분석, 실행 계획 수립 (`gpt-4o-mini`)
2. **Researcher** — arXiv / Semantic Scholar / Hugging Face 논문 수집 (`gpt-4o-mini`)
3. **Coder** — Methods 섹션 분석, PyTorch/TensorFlow 코드 생성 (`o4-mini`)
4. **Reviewer** — 생성 코드 vs 논문 이론 대조 검증, 피드백 작성 (`o4-mini`)
5. **Router** — 통과 시 사용자에게 전달, 낙제 시 Coder로 재전송 (최대 3회)

---

## Dev Commands

**Frontend**
```bash
cd FE
npm run dev        # 개발 서버 실행 (localhost:3000)
npm run build      # 프로덕션 빌드
npm run lint       # ESLint 실행
npm install        # 의존성 설치 (pull 후 항상 실행)
```

**Backend**
```bash
cd BE
uv sync                              # 의존성 설치 (pull 후 항상 실행)
uv run uvicorn main:app --reload     # 개발 서버 실행 (localhost:8000)
```

---

## API Design

- Frontend → Backend: **REST API** + **SSE (Server-Sent Events)**
- SSE로 에이전트 각 노드 진행 상황 실시간 스트리밍
- 모든 API 라우터는 `/api/v1` prefix 사용

**주요 엔드포인트**
```
GET  /health
GET  /api/v1/papers/search
POST /api/v1/agent/pdf
POST /api/v1/agent/search
POST /api/v1/agent/trend
```

---

## Coding Conventions

- **Python**: 타입 힌트 필수, async/await 사용
- **변수명**: `snake_case` (Python), `camelCase` (TypeScript)
- **컴포넌트**: 파일 하나에 컴포넌트 하나, `FE/components/`에만 작성
- **API 호출** (Frontend): 반드시 `FE/lib/api.ts`에서만
- **비즈니스 로직** (Backend): 반드시 `BE/services/`에서만, routers/에서 직접 작성 금지
- **에이전트 로직**: 반드시 `BE/agents/`에서만
- **커밋 메시지**: 영어로 작성 (아래 Git Convention 참고)
- **주석**: 한국어로 작성

---

## Git Convention

### Commit Message Format

형식: `{emoji} {Type}: {Description}`

예시:
```
✨ Feat: Add LangGraph planner node
🐛 Fix: Handle arXiv API rate limit error
♻️ Refactor: Separate agent state into typed fields
```

### Commit Types

| Emoji | Type | Description |
|-------|------|-------------|
| 🎉 | Start | Start new project |
| ✨ | Feat | Add new feature |
| 🐛 | Fix | Fix a bug |
| 🎨 | Design | Change UI/CSS |
| ♻️ | Refactor | Refactor code |
| 🔧 | Settings | Change configuration files |
| 🗃️ | Comment | Add or update comments |
| ➕ | Dependency | Add a dependency or plugin |
| 📝 | Docs | Update documentation |
| 🔀 | Merge | Merge branches |
| 🚀 | Deploy | Deploy to production |
| 🚚 | Rename | Rename or move files/folders |
| 🔥 | Remove | Delete files |
| ⏪️ | Revert | Revert to previous version |

### Branch Convention (GitHub Flow)

| Branch | Description |
|--------|-------------|
| `main` | Production-ready branch. Always deployable. |
| `develop` | Active development branch |
| `feature/{description}` | New feature branch (e.g. `feature/add-planner-node`) |

### Flow

1. Create `feature` branch from `develop`
2. Commit with proper message format
3. Push and create Pull Request (base: `develop`)
4. Merge into `develop` after review
5. Merge `develop` into `main` at deploy time

```bash
# 새 기능 브랜치 생성
git checkout -b feature/기능명

# 작업 후 커밋 & 푸시
git add .
git commit -m "✨ Feat: Add new feature"
git push origin feature/기능명

# GitHub에서 PR 생성
# base: develop ← compare: feature/기능명
```

### Git Rules

- **Commit messages must be written in English**
- 기능 완료 후 커밋까지는 자동으로 진행
- push는 항상 사용자 확인 후 진행
- Never commit directly to `main`
- Never hardcode API keys
- **로컬 `git merge`로 브랜치를 직접 병합하지 않는다** — 브랜치 병합은 GitHub PR 또는 웹 UI 기반 머지를 우선한다. 사용자가 명시적으로 로컬 머지를 요청한 경우에만 진행한다.

---

## Working Rules

- **코드 작성 전 반드시 계획을 먼저 세우고 사용자 확인 후 진행**
- 계획에는 변경할 파일, 변경 내용, 예상 동작 방식을 포함할 것

---

## Environment Variables

**Root `.env`**
```
OPENAI_API_KEY=
HUGGINGFACE_TOKEN=
LANGCHAIN_API_KEY=
```

**BE `.env`**
```
OPENAI_API_KEY=
DATABASE_URL=
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

**FE `.env.local`**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_DATA=false
```

---

## Important Rules

- `.env` 파일 절대 커밋 금지
- API 키 하드코딩 절대 금지 — 환경변수로만 관리
- `settings.local.json` 커밋 금지 (개인 설정)
- 새 에이전트 코드는 반드시 `BE/agents/`에 작성
- pull 후 Python은 `uv sync`, Node는 `npm install` 항상 실행