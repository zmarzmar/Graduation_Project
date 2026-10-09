#!/usr/bin/env bash
# 새 백엔드 컨테이너가 트래픽을 받아도 되는지 판단한다. deploy.sh가 source해서 쓴다.
#
# 한 번의 성공으로 전환하지 않는다 — 막 뜬 프로세스는 /health에 한두 번 답한 뒤 한동안 답하지 못할 수 있다
# (2026-10-07 배포에서 관측: 두 번 답한 뒤 39초 동안 응답 기록이 없었고, 그 사이 공개 요청이 실패했다).
# 연속으로 성공해야 준비됐다고 보고, 중간에 한 번이라도 실패하면 처음부터 다시 센다.
# 이것은 너무 일찍 전환할 위험을 줄이는 조치다. 통과한 뒤에 멈추지 않는다는 보장은 아니다.

# 한도는 모두 명시한다. 환경변수로 바꿀 수 있다.
READY_CONSECUTIVE="${READY_CONSECUTIVE:-10}"                        # 연속 성공 횟수
READY_INTERVAL_SECONDS="${READY_INTERVAL_SECONDS:-2}"               # 확인 간격 → 기본값이면 최소 18초 동안 안정적이어야 한다
READY_REQUEST_TIMEOUT_SECONDS="${READY_REQUEST_TIMEOUT_SECONDS:-2}" # 요청 하나의 제한 — 넘으면 실패로 센다
READY_DEADLINE_SECONDS="${READY_DEADLINE_SECONDS:-300}"             # 전체 대기 한도
READY_PROGRESS_SECONDS="${READY_PROGRESS_SECONDS:-20}"              # 기다리는 동안 진행 상황을 찍는 간격

# wait_until_ready <url>
# 준비되면 0, 전체 대기 한도 안에 조건을 채우지 못하면 1을 반환한다 (마지막 실패 원인을 출력한다).
wait_until_ready() {
  local url="${1}"
  local started="${SECONDS}" last_progress="${SECONDS}"
  local streak=0 attempts=0 failures=0 resets=0
  local last_error="(no request failed)" error

  while (( SECONDS - started < READY_DEADLINE_SECONDS )); do
    attempts=$((attempts + 1))
    if error="$(curl --connect-timeout "${READY_REQUEST_TIMEOUT_SECONDS}" --max-time "${READY_REQUEST_TIMEOUT_SECONDS}" \
                  -fsS -o /dev/null "${url}" 2>&1)"; then
      streak=$((streak + 1))
      if (( streak == 1 )); then
        echo "  first success after $((SECONDS - started))s (attempt ${attempts}); need ${READY_CONSECUTIVE} in a row"
      fi
      if (( streak >= READY_CONSECUTIVE )); then
        echo "  ready: ${streak} consecutive successes after $((SECONDS - started))s" \
             "(${attempts} attempts, ${failures} failed, streak reset ${resets} time(s))"
        return 0
      fi
    else
      failures=$((failures + 1))
      last_error="${error:-curl failed without a message}"
      if (( streak > 0 )); then
        resets=$((resets + 1))
        echo "  failed after ${streak} success(es), counting again from zero: ${last_error}"
      fi
      streak=0
    fi

    # 실패마다 찍지 않는다 — 앱이 뜨기 전의 실패는 정상적인 대기다
    if (( SECONDS - last_progress >= READY_PROGRESS_SECONDS )); then
      last_progress="${SECONDS}"
      echo "  waiting: $((SECONDS - started))s elapsed, ${attempts} attempts, ${streak}/${READY_CONSECUTIVE} in a row"
    fi
    sleep "${READY_INTERVAL_SECONDS}"
  done

  echo "  NOT ready within ${READY_DEADLINE_SECONDS}s: ${attempts} attempts, ${failures} failed," \
       "streak reset ${resets} time(s), ended at ${streak}/${READY_CONSECUTIVE} in a row"
  echo "  last error: ${last_error}"
  return 1
}
