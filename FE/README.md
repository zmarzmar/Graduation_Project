# AI Research Analyst — Frontend

Next.js 기반 프론트엔드. 사용자가 3가지 모드로 에이전트와 상호작용하는 웹 인터페이스.

---

## Features

- **PDF Upload** — 논문 PDF 업로드 → 에이전트 파이프라인 자동 실행
- **Keyword Search** — 키워드 입력 → arXiv + Semantic Scholar 논문 검색 및 분석
- **Trend Briefing** — 최신 트렌드 논문 Top N 요약 리포트 생성
- **실시간 스트리밍** — SSE로 에이전트 각 노드 진행 상황 실시간 표시

---

## Tech Stack

- **Framework**: Next.js (TypeScript, strict mode)
- **Styling**: Tailwind CSS, shadcn/ui
- **State**: Zustand (클라이언트), React Query (서버)

---

## Getting Started

```bash
npm install
npm run dev  # localhost:3000
```

---

## Environment Variables

```
# 백엔드 origin만 입력한다. /api/v1은 FE/lib/api.ts에서 자동으로 붙인다.
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_DATA=false
```
