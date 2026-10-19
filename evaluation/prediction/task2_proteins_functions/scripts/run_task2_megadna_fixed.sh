#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_KEYS="${MODEL_KEYS:-megadna}"
RESULTS_ROOT="${RESULTS_ROOT:-${TASK_DIR}/results}"
RUN_MODE="${RUN_MODE:-full}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
for MODEL_KEY in ${MODEL_KEYS}; do
  cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_t7.py"
       --model-key "${MODEL_KEY}" --results-dir "${RESULTS_ROOT}/${MODEL_KEY}"
       --data-dir "${DATA_DIR:-${TASK_DIR}/data/processed}"
       --device "${DEVICE:-cuda:0}" --run-mode "${RUN_MODE}" --n-jobs "${N_JOBS:-1}")
  if [[ -n "${CHECKPOINT:-}" ]]; then cmd+=(--checkpoint "${CHECKPOINT}"); fi
  if [[ -n "${MODEL_REPO:-}" ]]; then cmd+=(--model-repo "${MODEL_REPO}"); fi
  if [[ -n "${FEATURE_CACHE_DIR:-}" ]]; then cmd+=(--feature-cache-dir "${FEATURE_CACHE_DIR}"); fi
  if [[ -n "${GENES:-}" ]]; then cmd+=(--genes "${GENES}"); fi
  if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then cmd+=(--allow-download); fi
  if [[ "${REUSE_CACHED_FEATURES:-0}" == "1" ]]; then cmd+=(--reuse-cached-features); fi
  if [[ "${FORCE:-0}" == "1" ]]; then cmd+=(--force); fi
  "${cmd[@]}" "$@"
done
