# PaperPilot

> 논문을 읽고, 구현 초안을 만들고, 질문하는 AI 논문 분석 에이전트 — 졸업 프로젝트 (AI Research Analyst)

최신 AI 논문을 여러 출처에서 찾고, 논문 전문을 읽어 요약·강점과 한계·핵심 수식을 정리합니다. 이어서 논문을 바탕으로 **PyTorch 구현 초안**을 만들고, Coder와 Reviewer 에이전트가 검토와 수정을 최대 3회 반복합니다. 분석한 논문에는 질문할 수 있고, 답변 문장마다 인용한 원문 구절이 함께 표시됩니다.

**표현에 대해** — 만들어지는 코드는 논문의 "재현"이 아니라 **논문 기반 구현 초안**입니다. Reviewer의 판정은 LLM이 논문과 코드를 읽고 대조한 **"LLM 검토 통과"** 입니다. 실행 검사는 격리된 E2B 샌드박스에서 AI가 쓴 smoke_test를 작은 입력으로 돌려 오류 없이 끝나는지만 봅니다 — "실행 통과"가 논문대로 구현됐다는 뜻은 아닙니다. 평가에서 LLM 검토를 통과한 코드 중에도 실행하면 멈추는 코드가 있었습니다 ([평가](#평가)).

---

## 기능

| 기능 | 설명 |
|------|------|
| **PDF 업로드** | 업로드한 논문 PDF 전문을 바로 분석하고 구현 초안을 만듭니다 (검색 없음) |
| **키워드 검색** | Semantic Scholar · OpenAlex · arXiv에서 논문을 찾고, 고른 논문 한 편을 분석합니다. 전문은 arXiv에서만 받으며, 그 밖의 논문은 동의를 받아 초록으로 분석합니다 |
| **트렌드 브리핑** | Hugging Face · Semantic Scholar · arXiv 기반으로 최신 논문 흐름을 요약합니다 |
| **논문 Q&A** | 분석한 논문에 질문합니다 (로그인 사용자, 전문이 보관된 분석만). 답변 문장마다 인용한 원문 구절을 붙이고, 확인된 출처가 없으면 답하지 않습니다 |

논문 Q&A의 출처 확인은 **인용문이 그 구절에 실제로 있는지**까지입니다. 인용문이 답변 문장을 뒷받침하는지는 실행 중에 확인하지 않습니다.

---

## 구조

```
Frontend (Next.js, Vercel) ──REST / SSE──▶ Backend (FastAPI, Oracle Cloud · Docker · nginx blue/green)
                                              │
                                              ├─ 분석 그래프   Analyzer → Coder ⇄ Reviewer (최대 3회)
                                              ├─ 검색 그래프   Planner → Researcher → TrendAnalyzer
                                              ├─ 논문 Q&A     구절 검색 → 답변 생성 → 인용 확인
                                              │
                                              ├─ PostgreSQL 16  사용자 · 분석 기록 · 논문 원문
                                              └─ ChromaDB       논문 구절 벡터 색인 (원문에서 다시 만들 수 있는 파생 데이터)
```

- Coder와 Reviewer는 같은 논문 전문을 봅니다. 본문은 자르지 않고, 모델 한도(10만 토큰)를 넘으면 거부합니다
- 논문 본문이 들어가는 호출(Analyzer·Coder·Reviewer·논문 Q&A)은 호출 전에 입력 + 출력 예산이 컨텍스트 한도 안인지 실제 토크나이저로 확인합니다
- PDF는 arXiv 호스트에서만 받습니다 (리다이렉트마다 다시 검사)

## 기술 스택

- **Frontend**: Next.js 16, TypeScript, Tailwind CSS, Zustand, shadcn/ui, KaTeX
- **Backend**: FastAPI (Python 3.11), SQLAlchemy(async) + asyncpg, Alembic
- **AI**: LangGraph, OpenAI API — Analyzer·Planner·TrendAnalyzer·Q&A `gpt-4o-mini`, Coder·Reviewer `gpt-5.4-mini`(추론 강도 medium), 임베딩 `text-embedding-3-small`
- **DB**: PostgreSQL 16, ChromaDB 1.5.5
- **도구**: uv, npm, Docker Compose, GitHub Actions

---

## 평가

모두 논문 수가 적고 실행 횟수가 1회라, 경향을 보는 자료이지 일반적인 성능 주장이 아닙니다.

### 자기수정 루프 (2026-11-01)

Coder·Reviewer 모델보다 나중에 나온 arXiv 논문 10편을 각 1회 분석하고, 생성 전에 확정한 체크리스트(74항목, 논문의 절·식과 원문 인용 포함)로 회차별 코드를 판정했습니다.

- 체크리스트 충족: 1회차 **63/74**, 최종 **64/74**
- 수정 회차가 돈 논문은 2편이었고, 그중 Reviewer의 실제 판단으로 돈 것은 1편입니다. 그 1편에서는 Reviewer가 체크리스트가 잡은 결함을 정확히 지적했고 다음 회차에 고쳐졌습니다 (7/8 → 8/8)
- Reviewer가 1회차에 통과시킨 8편 중 5편에 체크리스트 미충족이 있었고, 그중 2편은 실행하면 멈추는 코드였습니다 (코드를 읽어 판단)
- 문법·import 정적 검사는 12개 코드를 모두 통과시켜 이 문제들을 걸러내지 못했습니다

한계: 논문당 1회 실행 · 판정은 Claude와 보조 에이전트가 코드 줄을 근거로 한 잠정 채점 · 생성 코드는 실행하지 않음. 원자료와 항목별 판정: [`eval-results/2026-11-loop-eval/`](eval-results/2026-11-loop-eval/), 도구: [`BE/evals/loop_eval.py`](BE/evals/loop_eval.py)

### 논문 Q&A (2026-10)

질문 24개, 논문 2편 + 지시문을 심은 문서 1개, 1회 실행: 답이 있는 질문 13/13 (같은 날 앞선 실행은 12/13), 답이 없거나 다른 논문에만 답이 있는 질문의 거부 10/10, 검색 범위 24/24, 심어둔 지시문을 따른 경우 0/2. LLM 판정기의 "뒷받침됨" 수치는 정확성의 근거로 쓰지 않습니다 — 같은 판정기가 인용보다 넓게 쓰인 문장 4건 중 1건만 잡았습니다. 도구: [`BE/evals/qa_eval.py`](BE/evals/qa_eval.py)

---

## 알려진 한계

- 실행 검사는 AI가 쓴 smoke_test로 오류 없이 끝나는지만 봅니다 (처음 나온 테스트로 고정 — 테스트 자체에 결함이 있으면 판단하지 못합니다). Reviewer 기준이 관대하고 TODO를 문제로 보지 않습니다
- 논문 Q&A는 인용문이 원문에 있는지만 확인합니다
- 남은 문제와 결정 기록은 [`CLAUDE.md`](CLAUDE.md)에 있습니다

---

## 로컬 실행

필요한 것: Docker, Python 3.11 + [uv](https://docs.astral.sh/uv/), Node.js + npm, OpenAI API 키

```bash
# 1. DB (저장소 루트에서)
docker compose up -d postgres chromadb

# 2. Backend
cd BE
uv sync
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
CHROMA_HOST=localhost
CHROMA_PORT=8001
EOF
uv run alembic upgrade head
uv run uvicorn main:app --reload          # http://localhost:8000

# 3. Frontend
cd ../FE
npm ci
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev                               # http://localhost:3000
```

`DATABASE_URL`을 비우면 로컬 개발 DB(`docker-compose.yml`)를 씁니다. 그 밖의 환경 변수는 [`CLAUDE.md`의 Environment Variables](CLAUDE.md#environment-variables)를 보세요. `npm run dev`가 tailwind 경로 문제로 깨지면 `npm run build && npm run start`로 띄웁니다.

### 테스트와 평가

```bash
cd BE
uv run python -m unittest discover -s tests -v       # DB·Chroma가 없으면 해당 테스트는 건너뜀
uv run python -m evals.qa_eval --out results.json    # 실제 API 비용 발생
uv run python -m evals.loop_eval --out results.json  # 실제 API 비용 발생, 논문 10편
cd ../FE && npm run lint && npm run build
```

---

## 프로젝트 구조

```
FE/   Next.js 앱 — app/(홈·마이페이지·관리자), components/agent(결과·Q&A), lib/api.ts(API 호출)
BE/   FastAPI — routers/, services/(비즈니스 로직), agents/(LangGraph 노드·Q&A), crud/, models/, alembic/, tests/, evals/
deploy/  blue/green 배포 스크립트와 nginx 설정
eval-results/  평가 원자료
```

개발 규칙, 설계 결정, Git 규칙은 [`CLAUDE.md`](CLAUDE.md)에 있습니다.
