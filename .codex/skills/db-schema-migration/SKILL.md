# db-schema-migration

## Purpose
- 모델, 스키마, CRUD, 마이그레이션 변경을 한 세트로 다룰 때 사용한다.

## Checklist
- `BE/models/`, `BE/schemas/`, `BE/crud/` 책임 분리를 유지한다.
- 마이그레이션 필요 여부와 적용 순서를 확인한다.
- 새 필드의 기본값, nullable, 기존 데이터 영향 범위를 점검한다.
- API 응답과 서비스 계층 영향 여부를 함께 본다.
- 로컬 개발 환경에서 필요한 설정과 의존성을 확인한다.
