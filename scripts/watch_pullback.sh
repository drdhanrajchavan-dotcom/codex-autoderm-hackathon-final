#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/.pulled_artifacts"
LOG_FILE="${LOG_DIR}/pullback.log"
INTERVAL_SECONDS="${PULLBACK_INTERVAL_SECONDS:-60}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S%z"
}

mkdir -p "${LOG_DIR}"

handle_sigint() {
  echo "[$(timestamp)] watch_pullback received SIGINT; exiting" | tee -a "${LOG_FILE}"
  exit 0
}

trap handle_sigint INT

echo "[$(timestamp)] watch_pullback start interval=${INTERVAL_SECONDS}s" | tee -a "${LOG_FILE}"

while true; do
  echo "[$(timestamp)] watch_pullback cycle start" | tee -a "${LOG_FILE}"

  set +e
  "${SCRIPT_DIR}/pullback.sh"
  status=$?
  set -e

  echo "[$(timestamp)] watch_pullback cycle exit status=${status}" | tee -a "${LOG_FILE}"
  sleep "${INTERVAL_SECONDS}" &
  wait $!
done
