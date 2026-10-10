#!/usr/bin/env bash
# readiness.sh의 준비 판정을 외부 서비스 없이 검증한다. curl을 가짜 함수로 바꿔 응답 순서를 정해 준다.
# 실행: bash deploy/test_readiness.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export READY_CONSECUTIVE=3 READY_INTERVAL_SECONDS=0 READY_REQUEST_TIMEOUT_SECONDS=2
export READY_DEADLINE_SECONDS=2 READY_PROGRESS_SECONDS=1
source ./readiness.sh

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# 가짜 curl: SCRIPT의 글자를 호출 순서대로 쓰고, 다 쓰면 처음부터 되풀이한다.
# S=성공, F=연결 실패, T=시간 초과, L=2초 뒤에야 도착하는 성공
# wait_until_ready는 curl을 $(...) 안에서 부르므로 호출 횟수는 파일에 적는다.
curl() {
  local n step
  n="$(cat "${WORK}/calls")"
  echo $((n + 1)) > "${WORK}/calls"
  echo "$*" > "${WORK}/args"
  step="${SCRIPT:$((n % ${#SCRIPT})):1}"
  case "${step}" in
    S) return 0 ;;
    L) sleep 2; return 0 ;;
    T) echo "curl: (28) Operation timed out" >&2; return 28 ;;
    *) echo "curl: (56) Recv failure: Connection reset by peer" >&2; return 56 ;;
  esac
}

failed=0
# check <이름> <응답 순서> <기대 종료 코드> <기대 호출 횟수 또는 -> <출력에 있어야 할 문자열>
check() {
  local name="${1}" expect_status="${3}" expect_calls="${4}" expect_text="${5}" status=0 output calls
  SCRIPT="${2}"
  echo 0 > "${WORK}/calls"
  output="$(wait_until_ready "http://127.0.0.1:1/health" 2>&1)" || status=$?
  calls="$(cat "${WORK}/calls")"
  if [[ "${status}" -ne "${expect_status}" ]] || { [[ "${expect_calls}" != "-" ]] && [[ "${calls}" -ne "${expect_calls}" ]]; } \
     || [[ "${output}" != *"${expect_text}"* ]]; then
    echo "FAIL ${name}: status=${status} (want ${expect_status}) calls=${calls} (want ${expect_calls}), output:"
    echo "${output}" | sed 's/^/    /'
    failed=1
  else
    echo "ok   ${name} (status=${status}, calls=${calls})"
  fi
}

# 앱이 뜨기 전의 실패는 기다리고, 뜬 뒤 연속 3회 성공하면 준비
check "starts late, then stable"            "FFFFSSS"  0 7 "ready: 3 consecutive successes"
# 성공 → 실패 → 회복: 실패가 연속 횟수를 0으로 되돌린다 — 성공이 모두 4번이어도 5번째 호출에서는 준비가 아니다
check "success, failure, recovery"          "SSFSSS"   0 6 "counting again from zero"
# 시간 초과도 실패다 (느리게 답하는 프로세스로 전환하지 않는다)
check "a timed-out request resets too"      "SSTSSS"   0 6 "Operation timed out"
# 전체 대기 한도: 끝내 뜨지 않으면 실패하고 마지막 원인을 남긴다
check "never comes up"                      "F"        1 - "last error: curl: (56) Recv failure"
# 답했다 멈췄다를 되풀이하면(성공 2번, 실패 1번이 끝없이 이어진다) 성공이 아무리 많아도 준비가 아니다
check "flapping never reaches the streak"   "SSF"      1 - "NOT ready within 2s"
# 한 번 성공한 것만으로는 전환하지 않는다 (예전 동작과의 차이) — 성공과 실패가 번갈아 온다
check "a single success is not enough"      "SF"       1 - "NOT ready within 2s"

# 전체 대기 한도는 응답이 도착한 시각으로도 지킨다: 한도(1초) 안에 보낸 요청이 2초 뒤에 성공해도 준비가 아니다
READY_CONSECUTIVE=1 READY_DEADLINE_SECONDS=1
check "a success that arrives after the limit is not counted" "L" 1 1 "arrived after the 1s limit"
# 같은 조건에서 제때 온 성공은 준비다 (위 검사가 한도 때문이지 다른 이유로 실패한 것이 아님을 확인)
check "a success inside the limit still counts"                "S" 0 1 "ready: 1 consecutive successes"
READY_CONSECUTIVE=3 READY_DEADLINE_SECONDS=2

# 요청 하나의 제한 시간이 curl에 실제로 전달된다
if [[ "$(cat "${WORK}/args")" != *"--max-time 2"* ]] || [[ "$(cat "${WORK}/args")" != *"--connect-timeout 2"* ]]; then
  echo "FAIL request timeout is not passed to curl: $(cat "${WORK}/args")"
  failed=1
else
  echo "ok   request timeout is passed to curl"
fi

# ── deploy.sh 전체 흐름: 준비되지 않으면 전환하지 않는다 ────────────────────────────
# docker·sudo·curl을 PATH의 가짜 실행 파일로 바꾸고 임시 디렉터리에서 deploy.sh를 실제로 돌린다.
STUBS="${WORK}/bin"
mkdir -p "${STUBS}"
cat > "${STUBS}/docker" <<'STUB'
#!/usr/bin/env bash
echo "docker $*" >> "${CALLS}"
STUB
cat > "${STUBS}/sudo" <<'STUB'
#!/usr/bin/env bash
echo "sudo $*" >> "${CALLS}"
STUB
cat > "${STUBS}/curl" <<'STUB'
#!/usr/bin/env bash
if [[ "${HEALTH}" == "up" ]]; then exit 0; fi
echo "curl: (56) Recv failure: Connection reset by peer" >&2
exit 56
STUB
chmod +x "${STUBS}"/*

# run_deploy <새 컨테이너 상태 up|down> → 종료 코드를 출력하고, 호출 기록과 상태 파일을 남긴다
run_deploy() {
  local app="${WORK}/app-${1}"
  mkdir -p "${app}/deploy/runtime"
  echo "green" > "${app}/deploy/runtime/backend_active"
  echo "server 127.0.0.1:18002;" > "${app}/deploy/runtime/backend_upstream.conf"
  : > "${WORK}/calls-${1}"
  local status=0
  PATH="${STUBS}:${PATH}" CALLS="${WORK}/calls-${1}" HEALTH="${1}" APP_DIR="${app}" \
    bash ./deploy.sh sha-test > "${WORK}/out-${1}" 2>&1 || status=$?
  echo "${status}"
}

expect() {
  if eval "${2}"; then echo "ok   ${1}"; else echo "FAIL ${1}"; failed=1; fi
}

status="$(run_deploy down)"
expect "not ready: the deploy fails"                         '[[ "${status}" -eq 1 ]]'
expect "not ready: nginx is not reloaded"                    '! grep -q "sudo" "${WORK}/calls-down"'
expect "not ready: the upstream still points at the old one" 'grep -q "18002" "${WORK}/app-down/deploy/runtime/backend_upstream.conf"'
expect "not ready: the active color is unchanged"            '[[ "$(cat "${WORK}/app-down/deploy/runtime/backend_active")" == "green" ]]'
expect "not ready: the old container is not stopped"         '! grep -q "stop.*backend_green" "${WORK}/calls-down"'
expect "not ready: the new container is stopped"             'grep -q "stop.*backend_blue" "${WORK}/calls-down"'
expect "not ready: the last error is in the log"             'grep -q "last error: curl: (56)" "${WORK}/out-down"'

status="$(run_deploy up)"
expect "ready: the deploy succeeds"                          '[[ "${status}" -eq 0 ]]'
expect "ready: the upstream points at the new one"           'grep -q "18001" "${WORK}/app-up/deploy/runtime/backend_upstream.conf"'
expect "ready: nginx is reloaded after the config test"      'grep -q "sudo nginx -t" "${WORK}/calls-up" && grep -q "sudo systemctl reload nginx" "${WORK}/calls-up"'
expect "ready: the active color is switched"                 '[[ "$(cat "${WORK}/app-up/deploy/runtime/backend_active")" == "blue" ]]'
expect "ready: the old container is stopped after the switch" 'grep -q "stop.*backend_green" "${WORK}/calls-up"'

exit "${failed}"
