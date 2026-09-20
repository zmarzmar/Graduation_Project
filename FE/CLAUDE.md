# FE/CLAUDE.md

프런트엔드 전용 안내. 프로젝트 구조, 현재 상태와 이어서 할 일(논문 Q&A 화면 계획 포함), Git 규칙은 **루트 `CLAUDE.md`** 에 있다 — 여기에는 FE에서만 필요한 것만 적는다.

Next.js 16 (App Router, TypeScript strict), Tailwind CSS, shadcn/ui, Zustand. 서버 상태 라이브러리는 쓰지 않는다 (React Query는 미사용이라 제거했다).

---

## 어디에 무엇을 쓰는가

| 위치 | 내용 |
|------|------|
| `app/` | 페이지 (`page.tsx` 홈, `mypage/`, `admin/`) |
| `components/agent/` | 에이전트 화면 컴포넌트. **파일 하나에 컴포넌트 하나** |
| `components/ui/` | shadcn/ui — 직접 고치지 않는다 |
| `lib/api.ts` | **모든 API 호출.** 컴포넌트에서 직접 fetch하지 않는다. 인증이 필요한 호출은 이 파일의 인증 fetch 헬퍼를 쓴다 |
| `lib/hooks/` | `useAgentStream`(SSE 처리), `useAuth` |
| `lib/types/agent-run.ts` | 에이전트 이벤트·결과 타입. BE 응답에 필드를 추가하면 여기도 함께 고친다 |
| `store/` | Zustand — `analysis-store`(모드별 스트림 상태), `auth-store` |

---

## 규칙

- 모든 props와 API 응답에 타입. 변수명 camelCase, 주석은 한국어, 스타일은 Tailwind만
- **SSE는 `EventSource`가 아니라 `fetch` + `ReadableStream`으로 읽는다** (POST와 인증 헤더가 필요하다). 처리는 `useAgentStream`에 있다
- **상태와 그 상태를 제어하는 핸들의 수명을 맞춘다.** 실행 상태는 store에 있어 페이지를 나가도 유지된다 — 취소 핸들(`AbortController`)도 같은 수명(모듈 수준 Map)에 둔다. 컴포넌트 ref에 두면 돌아왔을 때 취소할 수 없다
- 같은 모드의 스트림은 동시에 하나만. 실행 중이면 새로 시작하지 않는다
- 늦게 도착한 응답이 화면을 덮지 않게 한다 — 요청을 취소하고, 반영하기 전에 아직 같은 대상(분석·사용자)인지 확인한다
- FastAPI 오류는 `{"detail": "..."}` 형식이다. 사용자에게는 `detail`만 보여준다
- 분석 결과는 브라우저에 저장하지 않는다 (localStorage에는 인증 토큰과 오늘의 키워드 캐시만)
- 로그인 사용자의 분석은 원문이 서버에 보관된다 — `DocumentStorageNotice`가 안내한다. 보관 범위가 바뀌면 문구도 함께 고친다
- 긴 페이지 파일은 줄 수만 보고 쪼개지 않는다. 독립적인 역할이나 재사용 필요가 분명한 부분만 분리한다

---

## 실행과 검증

```bash
npm ci              # pull·브랜치 전환 후
npm run dev         # 홈 디렉터리의 lockfile 때문에 tailwind를 못 찾고 깨질 수 있다 → 아래로 우회
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run build && npm run start
npm run lint
```

- `NEXT_PUBLIC_API_URL`에는 백엔드 origin만 넣는다. `/api/v1`은 `lib/api.ts`가 붙인다
- 테스트 프레임워크가 없다. 동작은 브라우저에서 직접 확인한다 — 로컬 BE를 띄우고, 게스트는 1회만 실행할 수 있으므로 반복 확인은 로그인 상태로 한다
- CI는 lint + build만 돈다 (`FE/**` 변경 시)
