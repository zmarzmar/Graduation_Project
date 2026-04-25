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

echo "Waiting for backend_${NEXT_COLOR} health check..."
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${NEXT_PORT}/health" >/dev/null; then
    break
  fi

  if [[ "${i}" -eq 30 ]]; then
    echo "Health check failed for backend_${NEXT_COLOR}"
    exit 1
  fi

  sleep 2
done

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
  docker compose -f "${COMPOSE_FILE}" stop "backend_${CURRENT_COLOR}" || true
fi

echo "Switched active backend from ${CURRENT_COLOR}:${CURRENT_PORT:-not_running} to ${NEXT_COLOR}:${NEXT_PORT}"
