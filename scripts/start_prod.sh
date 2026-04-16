#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PID=""
WEB_PID=""

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
    kill "${WEB_PID}" 2>/dev/null || true
  fi
  wait "${API_PID}" "${WEB_PID}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

export PATH="${ROOT_DIR}/node_modules/.bin:/opt/venv/bin:${PATH}"

PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" && -x "/opt/venv/bin/python" ]]; then
  PYTHON_BIN="/opt/venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

API_HOST="${AUTODERM_API_HOST:-127.0.0.1}"
API_PORT="${AUTODERM_API_PORT:-8000}"
WEB_HOST="${AUTODERM_WEB_HOST:-0.0.0.0}"
WEB_PORT="${AUTODERM_WEB_PORT:-${PORT:-3000}}"

(
  cd "${ROOT_DIR}"
  exec "${PYTHON_BIN}" -m uvicorn src.api_server:app --host "${API_HOST}" --port "${API_PORT}"
) > >(sed -u 's/^/[api] /') 2>&1 &
API_PID="$!"

(
  cd "${ROOT_DIR}"
  exec next start web --hostname "${WEB_HOST}" --port "${WEB_PORT}"
) > >(sed -u 's/^/[web] /') 2>&1 &
WEB_PID="$!"

echo "[prod] API: http://${API_HOST}:${API_PORT}"
echo "[prod] App: http://${WEB_HOST}:${WEB_PORT}"

while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${WEB_PID}" 2>/dev/null; do
  sleep 1
done

wait "${API_PID}" "${WEB_PID}"
