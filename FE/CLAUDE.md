# FE/CLAUDE.md

This file provides frontend-specific guidance for the AI-arXiv Analyst project.

---

## Overview

Next.js 기반 프론트엔드. 사용자가 3가지 모드(PDF 업로드 / 키워드 검색 / 트렌드 브리핑)로 에이전트와 상호작용하는 웹 인터페이스.

---

## Tech Stack

- **Framework**: Next.js (TypeScript, strict mode)
- **Styling**: Tailwind CSS, shadcn/ui
- **State Management**: Zustand (클라이언트 상태), React Query (서버 상태)
- **Package Manager**: npm

---

## Folder Structure

```
FE/
├── app/                  # Next.js App Router 페이지
├── components/           # 재사용 가능한 UI 컴포넌트
├── lib/
│   └── api.ts            # API 호출 함수 전용 파일
├── hooks/                # 커스텀 훅
├── store/                # Zustand 스토어
└── types/                # TypeScript 타입 정의
```

---

## Dev Commands

```bash
npm run dev       # 개발 서버 실행 (localhost:3000)
npm run build     # 프로덕션 빌드
npm run lint      # ESLint 실행
npm install       # 의존성 설치 (pull 후 항상 실행)
```

---

## 3가지 모드 UI 구조

### 모드 1 - PDF 업로드
- PDF 파일 드래그 앤 드롭 업로드
- 업로드 후 자동으로 에이전트 파이프라인 실행
- 진행 상황 실시간 표시 (SSE 스트리밍)

### 모드 2 - 키워드 검색
- 키워드 입력 → 논문 목록 반환
- 논문 선택 → 분석 및 코드 재현 자동 진행
- 진행 상황 실시간 표시 (SSE 스트리밍)

### 모드 3 - 트렌드 브리핑
- 카테고리 선택 (LLM, Vision, RL 등)
- 인기 논문 Top N 목록 및 요약 리포트 표시

---

## Coding Conventions

- **컴포넌트**: 파일 하나에 컴포넌트 하나, `FE/components/` 폴더에만 작성
- **API 호출**: 반드시 `FE/lib/api.ts`에서만 호출. 컴포넌트에서 직접 fetch 금지
- **타입**: 모든 props와 API 응답에 TypeScript 타입 정의 필수
- **변수명**: camelCase
- **주석**: 한국어로 작성
- **스타일**: Tailwind CSS만 사용. 인라인 스타일 금지

---

## SSE 스트리밍 처리 방식

에이전트 진행 상황을 실시간으로 받아서 화면에 표시.

```typescript
// lib/api.ts에서만 SSE 연결 처리
const eventSource = new EventSource(`${API_URL}/api/v1/agent/search`)
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data)
  // Zustand store 업데이트
}
```

각 노드 진행 상황을 순서대로 표시:
Planner → Researcher → Coder → Reviewer → Router

---

## API 연결

- **Base URL**: `http://localhost:8000` (개발), `NEXT_PUBLIC_API_URL` 환경변수로 관리
- **API prefix**: `/api/v1`은 `FE/lib/api.ts`에서 자동으로 붙인다.
- **Mock 데이터**: `NEXT_PUBLIC_USE_MOCK_DATA=true` 시 목업 데이터 사용
- **실제 API 연결 시**: `NEXT_PUBLIC_USE_MOCK_DATA=false`로 변경

**엔드포인트**
```
POST /api/v1/agent/pdf      # PDF 업로드 모드
POST /api/v1/agent/search   # 키워드 검색 모드
POST /api/v1/agent/trend    # 트렌드 브리핑 모드
GET  /api/v1/papers/search  # 논문 검색
```

---

## Environment Variables

```
# 백엔드 origin만 입력한다. /api/v1은 FE/lib/api.ts에서 자동으로 붙인다.
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_DATA=false
```

---

## Important Rules

- API 호출은 반드시 `lib/api.ts`에서만
- 컴포넌트에서 직접 비즈니스 로직 작성 금지
- `admin-dashboard/` 폴더는 레거시 — 건드리지 말 것
- Mock 데이터에서 실제 API로 전환 시 `NEXT_PUBLIC_USE_MOCK_DATA=false`로만 변경
