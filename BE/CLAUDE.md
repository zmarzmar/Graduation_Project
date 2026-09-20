# BE/CLAUDE.md

백엔드 전용 안내. 프로젝트 구조, 현재 상태와 이어서 할 일, 설계 원칙, Git 규칙은 **루트 `CLAUDE.md`** 에 있다 — 여기에는 BE에서만 필요한 것만 적는다.

---

## 어디에 무엇을 쓰는가

| 위치 | 내용 |
|------|------|
| `routers/` | 요청·응답 형식, 상태 코드 매핑만. 쿼리와 로직을 직접 쓰지 않는다 |
| `services/` | 비즈니스 로직, 외부 API 호출, SSE 스트림 구성 |
| `agents/` | LLM을 쓰는 모든 로직 (노드, 그래프, 프롬프트, 토큰 예산, Q&A) |
| `crud/` | DB 쿼리. **commit하지 않는다** — 트랜잭션 경계는 호출자가 정한다 |
| `models/` | SQLAlchemy 테이블. 새 모델은 `models/__init__.py`에 등록해야 alembic이 본다 |
| `alembic/versions/` | 번호를 이어서 직접 작성 (`0009_…`). 추가만 하는 변경으로 만들고 `downgrade`도 쓴다 |
| `tests/` | `unittest` (pytest 아님) |

---

## 규칙

- 모든 함수에 타입 힌트, DB·외부 호출은 async/await
- 유저 데이터의 조회·삭제는 `user_id == current_user.id` + `is_deleted == False`. 분석 기록 삭제는 소프트 삭제다
- **DB 트랜잭션이나 advisory lock을 쥔 채로 LLM·임베딩·Chroma를 호출하지 않는다.** 상태 변경을 짧게 커밋한 뒤 호출한다
- SSE 엔드포인트는 DB 세션을 스트림 수명과 묶지 않는다 — `AsyncSessionLocal()`을 짧게 열고 닫는다
- 서버가 URL을 직접 방문하는 코드는 `_validate_pdf_url` 수준의 검증(허용 호스트, 매 홉 재검증)을 붙인다
- LLM에 논문 본문을 넣는 호출은 `agents/token_budget.py`의 `ensure_request_fits` / `ensure_complete`를 거친다. 본문을 조용히 자르지 않는다
- 모델 이름·한도 같은 조정값은 `core/config.py`에 둔다

---

## 테스트

```bash
uv run alembic upgrade head
uv run python -m unittest discover -s tests -v
REQUIRE_DB_TESTS=1 REQUIRE_CHROMA_TESTS=1 uv run python -m unittest discover -s tests   # CI와 같은 조건
```

- `ON CONFLICT`, `ON DELETE SET NULL`, 락처럼 mock으로 검증할 수 없는 동작은 **실제 Postgres**로 테스트한다
- 한 연결로 충분한 테스트는 바깥 트랜잭션에서 돌리고 롤백한다 (이때 `now()`는 트랜잭션 시작 시각으로 고정)
- 여러 연결이 필요한 동시 실행 테스트는 커밋한 뒤 직접 지운다 (`doc-test-…@example.com`)
- 테스트마다 `NullPool` 엔진을 만들고 서비스의 `AsyncSessionLocal`을 patch한다 — 전역 커넥션 풀은 이벤트 루프가 바뀌면 깨진다
- 실제 Chroma를 쓰는 테스트는 전용 컬렉션을 만들고 끝나면 지운다 (가짜 임베딩으로 `paper_chunks`에 쓰면 차원이 굳는다)
- 실제 OpenAI API를 쓰는 검증은 `evals/`에 두고 CI에서 돌리지 않는다
