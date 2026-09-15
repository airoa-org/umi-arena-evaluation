#!/usr/bin/env bash
# Checks the submission's runtime: against this image and starts the runner; --print-runtime only parses.
set -euo pipefail

SUBMISSION="${UMI_ARENA_SUBMISSION:-/submission}"
PORT="${UMI_ARENA_PORT:-8000}"
DEFAULT_RUNTIME=openpi

# Parsed in bash, not a sed pipeline: under pipefail a failing pipe would abort before any diagnostic.
parse_runtime() {
  local file="$1" line
  local key_pattern='^[[:space:]]*runtime[[:space:]]*:'
  local value_pattern="^[[:space:]]*runtime[[:space:]]*:[[:space:]]*([A-Za-z0-9_-]+|\"[A-Za-z0-9_-]+\"|'[A-Za-z0-9_-]+')([[:space:]]+#.*)?[[:space:]]*$"
  [[ -f "${file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" =~ ${key_pattern} ]]; then
      if [[ ! "${line}" =~ ${value_pattern} ]]; then
        echo "[entrypoint] invalid runtime declaration in ${file}; expected runtime: openpi or runtime: torch" >&2
        return 2
      fi
      local value="${BASH_REMATCH[1]}"
      value="${value//\"/}"
      printf '%s' "${value//\'/}"
      return 0
    fi
  done < "${file}"
}

RUNTIME="$(parse_runtime "${SUBMISSION}/umi_arena.yaml")"

if [[ "${1:-}" == "--print-runtime" ]]; then
  printf '%s\n' "${RUNTIME}"
  exit 0
fi

if [[ ! -d "${SUBMISSION}" ]]; then
  echo "[entrypoint] no submission mounted at ${SUBMISSION}" >&2
  echo "[entrypoint] mount one with: -v /path/to/submission:${SUBMISSION}:ro" >&2
  exit 2
fi

if [[ -z "${RUNTIME}" ]]; then
  RUNTIME="${DEFAULT_RUNTIME}"
  echo "[entrypoint] no runtime declared; defaulting to ${RUNTIME}"
fi

IMAGE_RUNTIME="${UMI_ARENA_RUNTIME:?UMI_ARENA_RUNTIME is not set; this is not a runtime image}"
if [[ "${RUNTIME}" != "${IMAGE_RUNTIME}" ]]; then
  echo "[entrypoint] submission runtime '${RUNTIME}' (declared, or the default) does not match this image, umi-arena-${IMAGE_RUNTIME}" >&2
  case "${RUNTIME}" in
    openpi|torch) echo "[entrypoint] run it with the umi-arena-${RUNTIME} image" >&2 ;;
    *) echo "[entrypoint] no image serves runtime '${RUNTIME}'; the runtimes are openpi and torch" >&2 ;;
  esac
  exit 2
fi

VENV="/opt/venv/${RUNTIME}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[entrypoint] image is missing executable ${VENV}/bin/python" >&2
  exit 2
fi
echo "[entrypoint] runtime=${RUNTIME} venv=${VENV} port=${PORT}"
exec "${VENV}/bin/python" /opt/umi_arena/runner.py \
  --port "${PORT}" \
  --adapter-dir "${SUBMISSION}" \
  --checkpoint-dir "${SUBMISSION}" \
  "$@"
