# backend-endpoint-change

## Purpose
- FastAPI 엔드포인트를 추가하거나 수정할 때 라우터와 서비스 경계를 유지한다.

## Checklist
- 요청/응답 스키마와 엔드포인트 계약을 먼저 확인한다.
- 라우터는 검증과 HTTP 입출력만 담당하도록 유지한다.
- 비즈니스 로직은 `BE/services/`에 배치한다.
- `/api/v1` prefix와 상태 코드, 응답 형태를 점검한다.
- 프론트 연동 영향이 있으면 함께 표시한다.
