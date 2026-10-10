#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/graduation-project}"
STATE_DIR="${STATE_DIR:-${APP_DIR}/deploy/runtime}"
ACTIVE_FILE="${ACTIVE_FILE:-${STATE_DIR}/backend_active}"
UPSTREAM_FILE="${UPSTREAM_FILE:-${STATE_DIR}/backend_upstream.conf}"
COMPOSE_FILE="${COMPOSE_FILE:-${APP_DIR}/docker-compose.prod.yml}"

BACKEND_BLUE_PORT="${BACKEND_BLUE_PORT:-18001}"
BACKEND_GREEN_PORT="${BACKEND_GREEN_PORT:-18002}"
BACKEND_IMAGE_TAG="${1:?usage: deploy.sh <image-tag>}"

# 준비 확인(wait_until_ready)과 그 한도
source "$(dirname "${BASH_SOURCE[0]}")/readiness.sh"

dump_backend_diagnostics() {
  local service_name="${1}"
  echo "Docker compose status for ${service_name}:"
  docker compose -f "${COMPOSE_FILE}" ps "${service_name}" || true
  echo "Recent logs for ${service_name}:"
  docker compose -f "${COMPOSE_FILE}" logs --tail=200 "${service_name}" || true
}

cleanup_old_backend_images() {
  echo "Cleaning old backend images, keeping latest 3 sha tags..."
  docker images "ghcr.io/*/graduation-project-be" \
    --format "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}" \
    | grep ":sha-" \
    | sort -r \
    | tail -n +4 \
    | cut -f2 \
    | xargs -r docker rmi || true
}

mkdir -p "${STATE_DIR}"
cd "${APP_DIR}"

if [[ ! -f "${ACTIVE_FILE}" ]]; then
  echo "none" > "${ACTIVE_FILE}"
fi

CURRENT_COLOR="$(cat "${ACTIVE_FILE}")"
if [[ "${CURRENT_COLOR}" == "blue" ]]; then
    NEXT_COLOR="green"
    NEXT_PORT="${BACKEND_GREEN_PORT}"
    CURRENT_PORT="${BACKEND_BLUE_PORT}"
elif [[ "${CURRENT_COLOR}" == "green" ]]; then
    NEXT_COLOR="blue"
    NEXT_PORT="${BACKEND_BLUE_PORT}"
    CURRENT_PORT="${BACKEND_GREEN_PORT}"
else
  NEXT_COLOR="blue"
  NEXT_PORT="${BACKEND_BLUE_PORT}"
  CURRENT_PORT=""
fi

export BACKEND_IMAGE_TAG

echo "Current color: ${CURRENT_COLOR}"
echo "Deploy target color: ${NEXT_COLOR}"
echo "Deploy image tag: ${BACKEND_IMAGE_TAG}"

docker compose -f "${COMPOSE_FILE}" pull postgres chromadb "backend_${NEXT_COLOR}"
docker compose -f "${COMPOSE_FILE}" up -d postgres chromadb "backend_${NEXT_COLOR}"

# 새 백엔드 컨테이너에서 마이그레이션을 실행한 뒤 nginx upstream을 전환한다.
# blue-green 배포 특성상 마이그레이션은 backward-compatible 해야 한다
# (기존 활성 컨테이너가 마이그레이션 중에도 여전히 요청을 처리하기 때문).
echo "Running alembic migrations on backend_${NEXT_COLOR}..."
for i in $(seq 1 10); do
  if docker compose -f "${COMPOSE_FILE}" exec -T "backend_${NEXT_COLOR}" uv run alembic upgrade head; then
    break
  fi

  if [[ "${i}" -eq 10 ]]; then
    echo "Alembic migration failed for backend_${NEXT_COLOR}"
    exit 1
  fi

  sleep 2
done

echo "Waiting for backend_${NEXT_COLOR} to be ready: ${READY_CONSECUTIVE} consecutive successes," \
     "${READY_REQUEST_TIMEOUT_SECONDS}s per request, every ${READY_INTERVAL_SECONDS}s, at most ${READY_DEADLINE_SECONDS}s"
if ! wait_until_ready "http://127.0.0.1:${NEXT_PORT}/health"; then
  # 여기까지는 nginx와 활성 색을 건드리지 않았다 — 트래픽은 기존 컨테이너가 계속 받는다
  echo "backend_${NEXT_COLOR} did not become ready. nginx was not changed; traffic stays on ${CURRENT_COLOR}."
  dump_backend_diagnostics "backend_${NEXT_COLOR}"
  # 준비되지 않은 컨테이너를 띄워 두지 않는다 (메모리가 작은 서버에서 기존 컨테이너와 자원을 다툰다)
  docker compose -f "${COMPOSE_FILE}" stop --timeout 20 "backend_${NEXT_COLOR}" || true
  exit 1
fi

cat > "${UPSTREAM_FILE}" <<EOF
upstream backend_upstream {
    server 127.0.0.1:${NEXT_PORT};
    keepalive 32;
}
EOF

sudo nginx -t
sudo systemctl reload nginx

echo "${NEXT_COLOR}" > "${ACTIVE_FILE}"
if [[ "${CURRENT_COLOR}" != "none" ]]; then
  docker compose -f "${COMPOSE_FILE}" stop --timeout 20 "backend_${CURRENT_COLOR}" || true
fi

echo "Switched active backend from ${CURRENT_COLOR}:${CURRENT_PORT:-not_running} to ${NEXT_COLOR}:${NEXT_PORT}"
cleanup_old_backend_images
echo "Deploy completed successfully"
exit 0
