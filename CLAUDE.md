# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Project Overview

**AI Research Analyst**(서비스명 PaperPilot)는 최신 AI 논문을 분석하는 Deep Research Agent 졸업 프로젝트다.

논문을 여러 출처(arXiv, Semantic Scholar, OpenAlex, Hugging Face)에서 수집하고, 논문 전문을 읽어 요약·리뷰·핵심 수식을 뽑고, **논문 기반 PyTorch 구현 초안**을 생성한 뒤 LangGraph의 **자기수정 루프**(Coder ↔ Reviewer)로 다듬는다. 분석한 논문에 질문하는 **논문 Q&A(RAG)** 를 추가하는 중이다.

표현 원칙: 현재 수준은 "코드 재현"이 아니라 **"논문 기반 구현 초안 생성"**, Reviewer의 판정은 "검증 통과"가 아니라 **"LLM 검토 통과"** 다. 코드를 실행해 검증하지 않는다.

---

## 🚧 현재 상태와 이어서 할 일 (2026-09-20 기준)

**새 세션은 여기부터 읽는다.** 아래는 결정이 끝난 내용이다 — 다시 논의하지 말고 이어서 진행한다.

### 상태

| PR | 내용 | 상태 |
|----|------|------|
| #33–#38 | 아래 "작업 기록" 참고 | 머지·운영 배포 완료 |
| **#39** `feature/paper-qa-rag-backend` | 논문 Q&A 백엔드 (RAG 3단계의 PR 2/3) | **열려 있음. CI 통과(101 tests). 머지 보류** |
| PR 3 (FE) | Q&A 화면 | 계획만 있음. #39 머지 후 시작 |

RAG 3단계는 PR 3개로 나눴다: ① 문서 보관(#38, 완료) → ② RAG 백엔드(#39) → ③ 프런트엔드.

### 1. #39를 머지하기 전에 해야 하는 것

**(a) 운영 Chroma 확인 — 서버 접속 권한이 있는 사람이 직접.** #39는 서버 이미지를 고정하고 볼륨 마운트 경로를 `/chroma/chroma` → `/data`로 바꿔서 컨테이너가 다시 만들어진다. 기존 마운트 경로는 서버가 실제로 쓰는 위치가 아니었으므로 **볼륨이 비어 있어도 컨테이너 내부 `/data`에 데이터가 있을 수 있다.** 마운트, 실제 저장 경로, 컬렉션 세 가지를 모두 확인한다 (명령은 #39 본문 맨 위). 데이터가 있으면 멈추고 보존 여부부터 정한다. 확인 없이 볼륨·컨테이너를 지우거나 초기화하지 않는다. 서버의 `/api/v2/version`은 이미지와 무관하게 `"1.0.0"`을 돌려주므로 버전 확인에 쓸 수 없다.

**(b) 인용문 정규화 범위를 더 좁힌다** (`BE/agents/qa.py`의 `_normalize`). 지금은 NFKC, casefold, 공백 제거, 줄 끝 하이픈, 단어 안 하이픈(양쪽 글자 2개 이상)을 정규화하고 부호·소수점·부등호는 보존한다. 그래도 수식 의미를 바꿀 수 있는 것이 남아 있다:
- casefold → `A`와 `a`가 같아진다 (행렬 A와 스칼라 a)
- NFKC → `²`와 `2`가 같아진다 (위첨자·아래첨자)
- 단어 안 하이픈 제거 → `alpha-beta`와 `alphabeta`가 같아진다

"의미를 보존한다"고 단정하지 말 것. 범위를 제한하고(예: 합자만 풀기, 대소문자 보존, 하이픈은 줄 끝에서 끊긴 경우만) 위 반례를 거부하는 테스트를 추가한다. 기존 반례 테스트는 `tests/test_rag.py`의 `CitationValidationTest`에 있다. 바꾼 뒤에는 `evals/qa_eval.py`를 다시 돌려 잘못된 거부가 얼마나 늘었는지 **수치로** 보고한다.

**(c) 고아 청크 정리의 한계를 명시하거나 주기 실행을 둔다.** `reconcile_orphan_chunks()`는 삭제 요청과 서버 기동 때만 돈다. 마지막 정리 이후에 늦게 도착한 청크는 다음 이벤트까지 Chroma에 남는다. **접근 차단(즉시)과 실제 삭제 완료(나중)는 별개**라는 점을 코드 주석과 PR에 명시하고, 확실한 정리가 필요하면 주기적인 재시도(예: lifespan에서 도는 백그라운드 루프)를 추가한다. 화면이나 문서에서 "원문이 완전히 삭제됐다"고 표현하지 않는다.

**(d) 코드 리뷰.** 위 (b)(c)를 포함해 #39 diff를 실제로 읽는 리뷰가 아직 없었다 (지금까지의 검토는 보고서 기준).

### 2. PR 3 (FE) 계획 — 확정된 내용

브랜치 `feature/paper-qa-frontend`, #39가 머지된 `main`에서 분기.

**BE 선행 변경 (PR 3에 포함):** 분석 완료 이벤트에 `analysis_id`와 `has_document`를 넣는다. 지금은 `_save_to_db`/`_save_analyze_to_db`가 아무것도 반환하지 않아 홈 결과 화면에서 질문할 수 없다. 저장을 `complete` 이벤트보다 먼저 하고 id를 반환하게 한다.
- `has_document`는 **실제 저장 결과**로 정한다 — 커밋된 분석 기록의 `document_id` 기준. PDF가 입력됐다는 사실만으로 true로 만들지 않는다 (원문 보관은 세이브포인트 안에서 실패할 수 있다).

**FE:**
- `FE/lib/api.ts`에 `askPaper(analysisId, question)` — 200 / 202 / 409를 구분해 반환
- `FE/components/agent/PaperQa.tsx` (신규) — 홈 결과(`ResultsPanel`)와 마이페이지 분석 상세 **양쪽에서 같은 컴포넌트**를 쓴다
- **Q&A 탭을 `analysis_id` 유무만으로 숨기지 않는다.** PDF·선택 논문 분석 결과에서는 항상 탭을 보여주고, 내용을 이 순서로 가른다: 로그인 여부(게스트 → 로그인 안내) → 저장 성공 여부(`analysis_id` 없음 → 저장 실패 안내) → 원문 보유 여부(`has_document=false` 또는 409 `no_document` → "논문을 다시 분석하면 질문할 수 있어요")
- **재시도 중 화면이 바뀌면 늦은 응답을 무시한다.** 타이머 정리만으로는 부족하다 — 요청을 `AbortController`로 취소하고, 응답을 반영하기 전에 "아직 같은 분석·같은 사용자인가"를 확인한다. 로그아웃하거나 다른 분석으로 이동했는데 이전 답변이 나타나면 안 된다
- **질문별 독립 답변임을 화면에 명확히 한다.** API는 질문 하나만 받으므로 "그 수식은?" 같은 후속 질문의 맥락을 모른다. 화면의 대화 기록(표시용)과 대화 문맥(모델에 전달)은 다른 것이다. 대화형 처리는 나중에
- **202 재시도 간격은 BE가 실제로 주는 값을 읽는다.** BE는 `Retry-After` 헤더와 본문의 `retry_after_seconds`를 둘 다 준다(값 동일). FE는 본문 필드 하나로 통일해서 읽는다 (CORS 설정이 `Retry-After`를 노출하지 않아 브라우저에서 읽을 수 없다). 재시도 횟수 상한을 둔다
- `answerable: false`는 오류가 아니라 정상 결과로 보이게 한다. `dropped_citations > 0`이면 "일부 출처를 확인하지 못했습니다"를 표시한다
- 출처 칩 `[p.5]`를 누르면 검증된 인용 구절을 펼친다. 원본 PDF 뷰어 이동은 범위 밖

### 3. 그 다음 단계

4. 같은 검색 기반으로 분석 결과(요약·수식·구현 설명)에 원문 근거 연결
5. 전문 vs 검색 컨텍스트의 코드 생성 비교 — **평가 실험으로만** (Coder·Reviewer의 기본 입력은 전문 유지)
6. 자기수정 루프 평가: 논문 10편, 최초 생성 vs 수정 후, 독립적인 실행 검사와 논문별 체크리스트. Reviewer 통과율은 보조 지표

### 4. 별도로 남겨둔 문제 (다른 PR에 섞지 말 것)

- Reviewer 프롬프트가 "통과 기준 (관대하게 적용)"이고 TODO를 문제로 보지 않는다. 화면 문구도 "LLM 검토 통과"로 바꿔야 한다
- 코드 실행 검증 없음. 도입한다면 1단계는 `ast.parse` + import **구문의 정적 확인**(실제 import는 코드를 실행하므로 금지), 실제 실행은 네트워크 없는 격리 컨테이너에서
- 배포 중 `/health`가 간헐적으로 타임아웃된다(blue/green인데 무중단이 아닌 구간). 원인 미확인 — 외부 관측만으로 nginx나 자원 부족으로 단정하지 말 것
- OpenAI TPM: 루프 3회면 o4-mini에 2분 남짓 동안 ~17만 토큰이 들어간다
- 초록 폴백 모달이 페이지 이동 후 다시 뜨지 않는다 (대기 중인 논문이 페이지 로컬 상태)
- 로컬 `npm run dev`가 tailwind 경로 문제로 깨진다(홈 디렉터리의 lockfile 때문에 Turbopack이 루트를 잘못 잡음). 우회: `npm run build && npm run start`
- `_extract_json`이 4개 노드에 복붙돼 있다 → `with_structured_output(PydanticModel)`로 교체. 모델명 하드코딩 6곳 → `config.py`
- 검색 중 Semantic Scholar 403, arXiv API 406이 관측됐다 (OpenAlex만 동작) — 일시적인지 확인 필요
- 미사용 import: `routers/mypage.py`(`HttpUrl`), `services/agent_service.py`(`io`)
- `AGENTS.md`와 `README.md`는 아직 옛 내용이다 (CLAUDE.md 3개만 2026-09-20에 갱신됨). #39가 머지되면 이 파일의 "(#39)" 표시도 정리한다

### 5. 로컬 환경 메모

- 로컬 개발 DB는 마이그레이션 **0009**(#39 브랜치)까지 적용돼 있다. `main`(head 0008)에서 alembic을 돌리면 "Can't locate revision 0009"가 난다 → #39 브랜치에서 `uv run alembic downgrade 0008` 하거나 #39 브랜치에서 작업한다
- 로컬 Chroma 컨테이너는 `chromadb/chroma:1.5.5`, 볼륨은 `/data`에 마운트돼 있다 (#39 기준)
- 로컬 DB에 검증용 계정 `canceltest`가 남아 있다
- FE `node_modules`는 브랜치를 바꾼 뒤 `npm ci`로 맞춘다 (#36에서 의존성이 빠졌다)

---

## 3 Core Modes

1. **PDF Upload Mode** — 논문 PDF를 업로드하면 **검색 없이** 업로드한 본문을 바로 분석하고 구현 초안을 만든다
2. **Keyword Search Mode** — 키워드로 논문을 검색(Semantic Scholar + OpenAlex + arXiv)하고, 사용자가 고른 논문 1편을 분석한다. 전문은 **arXiv에서만** 받는다 — 그 외 출처는 "초록만으로 분석할까요?" 확인으로 넘어간다
3. **Trend Briefing Mode** — Hugging Face + Semantic Scholar 기반 최신 트렌드 논문 요약 리포트

로그인 사용자가 분석한 논문은 추출 텍스트가 서버에 보관된다(논문 Q&A용). 게스트와 초록 기반 분석은 보관하지 않는다.

---

## Tech Stack

- **Frontend**: Next.js 16 (TypeScript, App Router), Tailwind CSS, Zustand, shadcn/ui, KaTeX
- **Backend**: FastAPI (Python 3.11), SQLAlchemy(async) + asyncpg, Alembic
- **AI Core**: LangGraph, OpenAI API
  - Planner, Analyzer, TrendAnalyzer, 논문 Q&A: `gpt-4o-mini` (컨텍스트 128k, 최대 출력 16,384)
  - Coder, Reviewer: `o4-mini` (컨텍스트 200k, 추론 토큰도 출력 예산에서 쓴다)
  - 임베딩(#39): `text-embedding-3-small`
- **Relational DB**: PostgreSQL 16 — 원본 데이터
- **Vector DB**: ChromaDB — 논문 Q&A의 파생 색인 (#39). `chromadb-client==1.5.5` + 서버 `chromadb/chroma:1.5.5`, 둘은 함께 올린다
- **Package Manager**: uv (Python), npm (Node.js)
- **배포**: BE는 Oracle 서버(Docker, blue/green, nginx), FE는 Vercel

---

## Project Structure

```
Graduation_Project/
├── CLAUDE.md
├── docker-compose.yml              # 로컬: postgres, chromadb
├── docker-compose.prod.yml         # 운영: postgres, chromadb, backend_blue/green
├── docker-compose.monitoring.yml
├── deploy/                         # deploy.sh(마이그레이션 → 헬스체크 → nginx 전환), nginx 설정
├── .github/workflows/
│   ├── ci.yml                      # FE lint + build (FE/** 변경 시)
│   └── backend.yml                 # BE 컴파일·마이그레이션 적용·테스트·Docker 빌드 → main이면 배포
├── FE/
│   ├── app/                        # page.tsx(홈), mypage/, admin/
│   ├── components/
│   │   ├── agent/                  # AgentPipeline, ResultsPanel, DocumentStorageNotice
│   │   ├── shared/                 # Header, Footer
│   │   └── ui/                     # shadcn/ui
│   ├── lib/
│   │   ├── api.ts                  # API 호출 전용 파일
│   │   ├── hooks/                  # useAgentStream, useAuth
│   │   └── types/agent-run.ts
│   └── store/                      # analysis-store, auth-store (Zustand)
└── BE/
    ├── main.py
    ├── core/                       # config.py(pydantic-settings), dependencies.py(DB 세션·인증)
    ├── routers/                    # agent, paper, auth, mypage, admin (+ qa: #39)
    ├── services/                   # agent_service, arxiv/semantic_scholar/openalex/keyword/auth/admin_service (+ rag_service: #39)
    ├── agents/                     # LangGraph 에이전트
    │   ├── graph.py, state.py
    │   ├── paper_context.py        # Coder·Reviewer가 공유하는 논문 원문 컨텍스트
    │   ├── token_budget.py         # 토큰 카운트, 호출 전 입력+출력 예산 검사
    │   ├── qa.py                   # (#39) 논문 Q&A 답변·출처 검증
    │   └── nodes/                  # planner, researcher, trend_analyzer, analyzer, coder, reviewer, router
    ├── models/                     # SQLAlchemy: user, paper, analysis, search_history, paper_document
    ├── crud/                       # DB 쿼리 (commit은 호출자가 한다)
    ├── schemas/                    # Pydantic DTO
    ├── alembic/versions/           # 0001 … 0008 (#39에서 0009)
    ├── tests/                      # unittest
    └── evals/                      # (#39) qa_eval.py — 실제 API를 쓰는 평가, CI에서 돌리지 않는다
```

---

## Agent Architecture

그래프가 두 개다 (`BE/agents/graph.py`).

```
검색 그래프 (search / trend)
  Planner → Researcher → (trend) TrendAnalyzer → END
                       → (search) END   ← 사용자가 논문을 고르면 분석 그래프로

분석 그래프 (pdf 업로드 / 선택한 논문 분석)
  Analyzer → Coder → Reviewer → Router ─ 통과 또는 3회 → END
                ↑                  │
                └──── 낙제 ────────┘
```

1. **Planner** — 검색 키워드와 계획 요약을 만든다. 출력은 `summary`, `search_keywords`뿐이다
2. **Researcher** — Semantic Scholar / OpenAlex / arXiv / Hugging Face에서 수집
3. **TrendAnalyzer** — 트렌드 논문 요약과 키워드
4. **Analyzer** — 논문 요약, 리뷰(강점·한계), 핵심 수식(LaTeX)
5. **Coder** — 논문 전문으로 PyTorch 구현 초안 생성, 2회차부터는 리뷰 피드백 반영
6. **Reviewer** — 생성 코드와 논문을 LLM이 대조해 피드백 작성 (코드를 실행하지 않는다)
7. **Router** — 통과 시 종료, 낙제 시 Coder로 (최대 3회)

---

## 설계 원칙 (2026-09-20에 확정 — 되돌리지 말 것)

**논문 본문과 토큰**
- Coder와 Reviewer는 **같은 전문**을 본다 — 논문 원문 부분은 `paper_source_context()` 한 곳에서 만든다. 통일하는 것은 원문 부분뿐이고 요약·수식·이전 코드는 노드별로 붙인다
- 본문을 **자르지 않는다.** `max_paper_tokens`(기본 10만, 가장 좁은 Analyzer 기준)를 넘으면 거부하고 사용자에게 알린다 (업로드 413, 다운로드 경로는 초록 폴백 제안)
- 토큰은 실제 토크나이저(tiktoken `o200k_base`)로 센다. 바이트 근사는 쓰지 않는다 — 수식 기호·숫자 표를 실제의 0.54배로 적게 잡는다. 토크나이저를 못 쓰면 추정으로 진행하지 않고 오류를 낸다. 인코딩 파일은 Docker 이미지에 미리 받아둔다
- 모든 노드는 API에 출력 한도를 보내고(`max_tokens`), **호출 직전마다** 전체 입력 + 출력 예산이 컨텍스트 한도 안인지 확인한다. 잘린 응답(`finish_reason: length`)은 결과로 쓰지 않는다
- PDF는 한 번만 파싱하고 **페이지별로** 보존한다. 빈 페이지도 남긴다 — 인덱스 + 1 = 원본 페이지 번호

**서버가 방문하는 URL**
- PDF는 `arxiv.org`, `www.arxiv.org`, `export.arxiv.org`에서만 받는다. https, 정확한 호스트 일치, 포트 443, userinfo 금지. 리다이렉트는 수동으로 최대 3회 따라가며 매 홉 재검증. URL 파싱은 요청을 보내는 `httpx.URL`로 한다

**논문 원문 보관 (#38)**
- Postgres(`paper_documents`)가 원본, Chroma는 다시 만들 수 있는 파생 색인. **접근 권한은 Postgres에서만 판정한다**: 본인의 삭제되지 않은 분석 기록이 가리키는 문서
- 중복 제거는 **사용자별** `(user_id, doc_hash)`. `doc_hash`는 실제로 저장하는 페이지 목록의 sha256 — "같은 논문"이 아니라 "동일한 추출 본문"의 기준. 사용자 간 공유는 하지 않는다
- 게스트는 보관하지 않는다 — 서버에 게스트를 구분할 수단이 없다 (1회 제한은 FE에서만 처리)
- 문서 저장과 정리는 사용자 단위 advisory lock으로 직렬화한다. **락이나 DB 트랜잭션을 쥔 채로 LLM·임베딩·Chroma를 호출하지 않는다**

**논문 Q&A / RAG (#39)**
- RAG는 **Q&A와 근거 연결**에 쓴다. 논문 한 편(~1만 토큰)은 컨텍스트에 통째로 들어가므로 Coder·Reviewer의 기본 입력을 검색 결과로 바꾸지 않는다
- 로그인 전용. 색인은 첫 질문 때 요청 안에서 만든다 (60초 제한). 다른 요청이 색인 중이면 202
- 색인 작업마다 `index_job_id`를 발급하고 청크에 job_id를 붙인다. 검색은 문서 + 현재 job + `ready`만. 완료·실패 기록은 현재 작업만 할 수 있다 → 오래된 작업이 새 색인을 훼손하지 못한다
- 삭제는 **툼스톤**: 마지막 활성 기록이 지워지면 pages를 즉시 비우고 `doc_hash`를 `purged:{id}`로 바꾼다. **삭제 중인 문서는 복구하지 않는다** — 같은 본문은 새 id의 새 문서가 된다. 순서는 purge_pending → Chroma 삭제 → 유예 후 재삭제 → 행 제거. Chroma가 실패하면 툼스톤이 남아 다시 시도한다
- 색인 유실은 "필터 전 결과 0건 + 실제 청크 0개"일 때만. 근거 부족은 정상적인 "근거를 확인하지 못함"
- 검색된 구절은 구분자로 감싼 **자료**로 전달한다 — 구절 안의 지시문은 따르지 않는다
- 출처: 인용 번호 검증 ≠ 근거 검증. 모델이 인용문을 그대로 옮기게 하고 서버가 그 구절에 실제로 있는지 확인한다. 유효한 출처가 0개면 모델의 답을 내보내지 않는다. 인용문이 답을 **뒷받침하는지**는 런타임에 확인하지 못한다 — 평가에서 본다
- 거리 임계값은 보정 질문에서 분리되지 않아 꺼두었다 (`qa_max_distance=1.0`)
- 클라이언트가 연결을 끊어도 핸들러는 **취소되지 않는다** (실제 uvicorn으로 확인). 색인은 끝까지 돌거나 시간 제한이 놓아준다

---

## Dev Commands

**Frontend**
```bash
cd FE
npm ci             # 의존성 설치 (pull·브랜치 전환 후)
npm run dev        # 개발 서버 (localhost:3000) — 깨지면 npm run build && npm run start
npm run build
npm run lint
```

**Backend**
```bash
cd BE
uv sync                                  # 의존성 설치 (pull 후 항상)
docker compose up -d postgres chromadb   # 저장소 루트에서
uv run alembic upgrade head              # 마이그레이션
uv run uvicorn main:app --reload         # 개발 서버 (localhost:8000)

# 테스트 (unittest, pytest 아님)
uv run python -m unittest discover -s tests -v
REQUIRE_DB_TESTS=1 REQUIRE_CHROMA_TESTS=1 uv run python -m unittest discover -s tests   # DB·Chroma에 못 붙으면 건너뛰지 않고 실패 (CI와 같음)

# 평가 (#39, 실제 API 비용 발생)
uv run python -m evals.qa_eval --out results.json
```

DB 테스트는 실제 Postgres에 붙는다. 롤백 격리 테스트는 흔적을 남기지 않고, 여러 연결이 필요한 테스트는 커밋한 뒤 직접 지운다 (`doc-test-…@example.com`).

---

## API Design

- Frontend → Backend: **REST API** + **SSE** (에이전트 각 노드의 진행 상황을 실시간 스트리밍)
- 모든 API 라우터는 `/api/v1` prefix

```
GET    /health
POST   /api/v1/agent/search                      # SSE — 논문 검색
POST   /api/v1/agent/analyze                     # SSE — 선택한 논문 1편 분석
POST   /api/v1/agent/pdf                         # SSE — PDF 업로드 분석 (413: 너무 긴 논문, 422: 추출 실패)
POST   /api/v1/agent/trend                       # SSE — 트렌드 브리핑
GET    /api/v1/papers/search
GET    /api/v1/papers/daily-keywords
POST   /api/v1/auth/register | /auth/login       GET /api/v1/auth/me
GET    /api/v1/mypage/me                         PATCH /api/v1/mypage/me
GET    /api/v1/mypage/search-history             DELETE …/search-history[/{id}]
GET    /api/v1/mypage/analysis-history[/{id}]    DELETE …/analysis-history[/{id}]   # 응답에 has_document
GET    /api/v1/admin/users | /admin/papers | /admin/system
POST   /api/v1/analyses/{analysis_id}/ask        # (#39) 논문 Q&A — 401 / 404 / 409 no_document / 202 색인 중 / 504
```

---

## Coding Conventions

- **Python**: 타입 힌트 필수, async/await 사용
- **변수명**: `snake_case` (Python), `camelCase` (TypeScript)
- **컴포넌트**: 파일 하나에 컴포넌트 하나, `FE/components/`에만 작성. 긴 페이지 파일은 줄 수만 보고 쪼개지 말고 독립적인 역할·재사용 필요가 분명한 부분만 분리한다
- **API 호출** (Frontend): 반드시 `FE/lib/api.ts`에서만
- **비즈니스 로직** (Backend): 반드시 `BE/services/`에서만, routers/에서 직접 작성 금지. DB 쿼리는 `BE/crud/`
- **에이전트 로직**: 반드시 `BE/agents/`에서만
- **소유권 검증**: 유저 데이터를 다루는 조회·삭제는 `user_id == current_user.id` + `is_deleted == False`
- **커밋 메시지**: 영어로 작성 (아래 Git Convention 참고)
- **주석**: 한국어로 작성

---

## Git Convention

### Commit Message Format

형식: `{emoji} {Type}: {Description}` — **아래 표에 있는 타입만 쓴다** (`Test` 같은 타입은 없다. 테스트 추가는 `Feat`/`Fix`에 포함한다)

```
✨ Feat: Add LangGraph planner node
🐛 Fix: Handle arXiv API rate limit error
♻️ Refactor: Separate agent state into typed fields
```

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
| `main` | Default branch. Production-ready and deployed to production. |
| `feature/{description}` | 새 기능·리팩터링·문서 (e.g. `feature/add-planner-node`) |
| `fix/{description}` | 버그 수정 (e.g. `fix/validate-pdf-download-url`) |

### Flow

1. `main`에서 브랜치 생성
2. 작업별로 커밋을 나눈다 (규칙에 맞는 메시지)
3. Push → Pull Request (base: `main`)
4. CI 확인. PR의 CI는 최신 `main`과 합친 상태로 돈다
5. 리뷰 후 **merge commit** 방식으로 머지
6. 머지 확인 후 로컬·원격 브랜치 삭제

### Git Rules

- **Commit messages must be written in English**
- 기능 완료 후 커밋까지는 자동으로 진행. **push는 항상 사용자 확인 후**
- Never commit directly to `main`. 모든 브랜치는 `main`에서 만든다. 장기 유지 브랜치는 `main`뿐
- **로컬 `git merge`로 병합하지 않는다** — **GitHub MCP (`mcp__github__create_pull_request` → `mcp__github__merge_pull_request`)** 가 기본. MCP를 못 쓸 때만 `gh` CLI나 웹 UI. 로컬 머지는 사용자가 명시적으로 요청한 경우에만
- 머지 전에 확인: CI 통과, 리뷰·인라인 코멘트, `mergeStateStatus`, **진행 중인 배포가 없는지**
- **배포**:
  - `main`에 `BE/**`, compose, deploy 파일이 바뀌어 푸시되면 **운영 배포**가 나간다. FE만 바뀌면 Vercel만 배포된다
  - `backend.yml`은 `cancel-in-progress: true`다 — **배포가 도는 중에 머지하면 blue/green 교체 도중의 배포가 취소된다.** 앞선 배포가 끝난 것을 확인하고 머지한다
  - `deploy` job은 `backend-ci`(마이그레이션 적용 + 테스트)를 통과해야 돈다
  - 서버 동기화는 `git pull` 대신 `fetch + reset --hard origin/main`
  - GitHub Actions는 `production` environment의 secret/variable 이름으로 배포 대상을 관리한다
  - `deploy/runtime/*` 같은 런타임 생성 파일은 추적하지 않는다

---

## Working Rules

- **코드 작성 전 반드시 계획을 먼저 세우고 사용자 확인 후 진행.** 계획에는 변경할 파일, 변경 내용, 예상 동작을 포함한다
- 사용자는 다른 AI 리뷰어의 검토 의견을 붙여넣어 **교차 검증**한다. 의견을 그대로 따르지도 무시하지도 말고 코드로 확인한 뒤 동의·반대를 근거와 함께 말한다
- **추정하지 말고 확인한다.** 이 프로젝트에서 실제로 틀렸던 추정들: "전문을 넣으면 놓칠 수 없다", "바이트/3 추정은 보수적이다", "연결이 끊기면 작업이 취소된다", "볼륨이 비었으면 데이터가 없다", "서버 버전은 `/version`으로 알 수 있다". 측정·재현·로그로 확인하고, 확인하지 못한 것은 확인하지 못했다고 쓴다
- 동시 실행 문제는 실제 DB 연결 여러 개로 재현해서 테스트한다. mock으로 검증할 수 없는 DB 동작(`ON CONFLICT`, `ON DELETE SET NULL`, 락)은 실제 Postgres로
- 기능을 줄이는 변경(예: arXiv 외 PDF 차단, PDF 모드의 관련 논문 제거)은 **의도적인 기능 축소**라고 PR에 명시한다
- 평가 결과는 표본 크기와 한계를 함께 적는다. 실패를 보고 코드를 고쳤다면 그 질문은 더 이상 "처음 보는" 질문이 아니다 → 새 질문을 추가한다
- 운영 서버에는 접속 권한이 없다. 서버에서 확인해야 하는 것은 명령을 적어 사용자에게 넘긴다

---

## Environment Variables

**BE `.env`**
```
OPENAI_API_KEY=
DATABASE_URL=                  # postgresql+asyncpg://…  기본값은 로컬 개발 DB
JWT_SECRET_KEY=                # DATABASE_URL이 로컬 기본값이 아니면 필수 (기본 시크릿이면 기동 거부)
SEMANTIC_SCHOLAR_API_KEY=      # 선택
OPENALEX_EMAIL=                # 선택 — polite pool
CHROMA_HOST=localhost
CHROMA_PORT=8001
CORS_ORIGINS=                  # 선택, 쉼표 구분
MAX_PAPER_TOKENS=100000        # 선택
```
`seed.py`는 `SEED_ADMIN_*`, `SEED_TEST_*`를 읽는다 (자격 증명을 코드에 두지 않는다).

**FE `.env.local`**
```
# 백엔드 origin만 입력한다. /api/v1은 FE/lib/api.ts에서 자동으로 붙인다.
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Important Rules

- `.env` 파일 절대 커밋 금지. API 키 하드코딩 금지 — 환경변수로만
- `settings.local.json` 커밋 금지 (개인 설정)
- 새 에이전트 코드는 반드시 `BE/agents/`에 작성
- pull 후 Python은 `uv sync`, Node는 `npm ci`
- ChromaDB를 제거하자고 다시 제안하지 말 것 — RAG 방향으로 가기로 결정했다

---

## 작업 기록

### 2026-09-20

오래된 아이디어를 점검하면서 시작해, 버그 수정 → 미사용 구성 제거 → 생성·검토 근거 통일 → RAG 도입 순으로 진행했다.

| PR | 내용 |
|----|------|
| #33 | **PDF 다운로드 URL 검증 (SSRF 차단).** 클라이언트가 준 `pdf_url`을 검증 없이 `follow_redirects=True`로 방문하고 있었다. arXiv 호스트만 허용, 매 홉 재검증. BE 테스트와 CI 테스트 단계 최초 도입 |
| #34 | **PDF 모드가 검색된 다른 논문을 "분석 논문"으로 표시하던 문제.** PDF 모드는 Planner·Researcher를 건너뛰고 분석 그래프로 바로 간다. 결과에 `uploaded_filename`. 텍스트 없는 PDF는 분석 전에 거부 |
| #35 | **페이지를 나갔다 오면 분석을 취소할 수 없던 문제.** `AbortController`를 컴포넌트 ref에서 모듈 수준 Map으로 옮겨 store와 수명을 맞췄다. 같은 모드의 중복 실행 방지 |
| #36 | FE 미사용 의존성 제거(`zod`, `date-fns`, `@tanstack/react-query`), Planner의 쓰이지 않는 출력(`focus_area`, `framework`)과 도달 불가 분기 제거 |
| #37 | **Reviewer가 논문 앞 8,000자만 보던 문제** → Coder와 같은 전문. PDF 페이지 보존, 길이 초과 시 자르지 않고 거부, 실제 토크나이저, 호출별 입력+출력 예산 |
| #38 | **논문 원문 보관** (`paper_documents`, 마이그레이션 0008). 사용자별 중복 제거, 마지막 활성 기록 삭제 시 문서 정리, 동시 저장·삭제 경쟁을 advisory lock으로 해결. CI에 Postgres 서비스와 마이그레이션 적용 단계. 원문 보관 안내 문구 |
| #39 (열림) | **논문 Q&A 백엔드.** 마이그레이션 0009, 청크 분할, 첫 질문 때 색인, 작업별 job id, 툼스톤 삭제, 고아 청크 대조 정리, 출처 검증, `chromadb-client` 교체와 서버 버전 고정, 평가 스크립트, CI에 Chroma 서비스 |

**결정:** ChromaDB 제거 → 보류하고 RAG로. RAG는 Q&A·근거 연결에, Coder·Reviewer는 전문 유지. Q&A는 로그인 전용, 첫 질문 때 색인, 사용자별 중복 제거.

**측정값 (참고):**
- 논문 길이(실제 토큰): Attention 10k / LoRA 26k / LLaMA 27k / GPT-3 64k / PaLM 83k — 87쪽짜리도 한도(10만) 안
- LoRA 논문 1회 실행: Reviewer가 앞 8,000자만 볼 때 1회차 통과(LLM 호출 3회, 입력 5.7만 토큰) → 전문을 볼 때 2번 낙제 후 3회차 통과(호출 7회, 입력 19.8만 토큰). 회당 입력 증가와 반복 증가가 함께 작용한 결과이고, **단일 실행 관측값**이지 품질 개선의 증명이 아니다
- Q&A 평가(질문 24개, 논문 2편 + 지시문을 심은 문서, 단일 실행): 답 있는 질문 11/13, 답 없는·다른 논문 질문 거부 10/10, 검색 범위 24/24, 심어둔 지시문 따름 0/2. 같은 질문이 실행마다 통과·실패가 갈렸다
