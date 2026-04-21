---
name: fullstack-contract-sync
description: FE와 BE 계약이 함께 바뀌는 작업에서 경로와 필드를 동기화하기 위한 체크리스트
---

# fullstack-contract-sync

## Purpose
- FE/BE 계약이 함께 바뀌는 작업에서 경로, 필드, 응답 형태를 동기화한다.

## Checklist
- 백엔드 엔드포인트, 필드명, 응답 형태 변경점을 먼저 정리한다.
- `BE/routers/`, `BE/services/`, `FE/lib/api.ts`를 함께 갱신한다.
- 프론트 소비 코드와 타입 정의를 같이 맞춘다.
- 스트리밍 계약 변경 시 소비 방식까지 함께 확인한다.
- 최소 한 번의 계약 검증 또는 스모크 테스트를 수행한다.
