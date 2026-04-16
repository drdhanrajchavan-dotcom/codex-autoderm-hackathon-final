#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
ARTIFACT_ROOT="${REPO_ROOT}/.pulled_artifacts"
RUNS_DEST="${ARTIFACT_ROOT}/runs"
LOG_FILE="${ARTIFACT_ROOT}/pullback.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S%z"
}

mkdir -p "${RUNS_DEST}"

if [[ ! -f "${ENV_FILE}" ]]; then
  {
    echo "[$(timestamp)] ERROR: ${ENV_FILE} not found. Create it from .env.example."
  } | tee -a "${LOG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${EC2_HOST:?EC2_HOST must be set in .env}"
: "${EC2_PROJECT_DIR:?EC2_PROJECT_DIR must be set in .env}"
: "${EC2_SSH_KEY:?EC2_SSH_KEY must be set in .env}"

if [[ ! -f "${EC2_SSH_KEY}" ]]; then
  {
    echo "[$(timestamp)] ERROR: EC2_SSH_KEY does not exist: ${EC2_SSH_KEY}"
  } | tee -a "${LOG_FILE}" >&2
  exit 2
fi

REMOTE_PROJECT_DIR="${EC2_PROJECT_DIR%/}"
REMOTE_RUNS="ubuntu@${EC2_HOST}:${REMOTE_PROJECT_DIR}/runs/"

{
  echo "[$(timestamp)] pullback start"
  echo "[$(timestamp)] remote: ${REMOTE_RUNS}"
  echo "[$(timestamp)] local: ${RUNS_DEST}/"
} | tee -a "${LOG_FILE}"

set +e
rsync -avz --update -e "ssh -i ${EC2_SSH_KEY}" "${REMOTE_RUNS}" "${RUNS_DEST}/" 2>&1 | tee -a "${LOG_FILE}"
rsync_status=${PIPESTATUS[0]}
set -e

if [[ ${rsync_status} -ne 0 ]]; then
  echo "[$(timestamp)] pullback failed status=${rsync_status}" | tee -a "${LOG_FILE}" >&2
  exit "${rsync_status}"
fi

best_count="$(find "${RUNS_DEST}" -path "*/weights/best.pt" -type f | wc -l | tr -d " ")"
echo "[$(timestamp)] pullback success status=0 best_pt_files=${best_count}" | tee -a "${LOG_FILE}"
