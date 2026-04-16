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

export PATH="${ROOT_DIR}/node_modules/.bin:${PATH}"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

(
  cd "${ROOT_DIR}"
  exec "${PYTHON_BIN}" -m uvicorn src.api_server:app --port 8000 --reload
) > >(sed -u 's/^/[api] /') 2>&1 &
API_PID="$!"

(
  cd "${ROOT_DIR}/web"
  exec npm run dev -- --port 3000
) > >(sed -u 's/^/[web] /') 2>&1 &
WEB_PID="$!"

echo "[dev] API: http://localhost:8000"
echo "[dev] App: http://localhost:3000"

while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${WEB_PID}" 2>/dev/null; do
  sleep 1
done

wait "${API_PID}" "${WEB_PID}"
