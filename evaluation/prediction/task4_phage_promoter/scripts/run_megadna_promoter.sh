#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
DATA_PATH="${DATA_PATH:-${TASK_DIR}/data/processed/phage_promoter_dataset2400_65bp.tsv}"
RUN_MODE="${RUN_MODE:-full}"
FORCE="${FORCE:-0}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
common=(--data_path "${DATA_PATH}" --device "${DEVICE}" --run_mode "${RUN_MODE}")
if [[ "${FORCE}" == "1" ]]; then common+=(--force); fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then common+=(--allow_download); fi
if [[ "${REUSE_CACHED_FEATURES:-0}" == "1" ]]; then common+=(--reuse_cached_features); fi
if [[ -n "${MODEL_REPO:-}" ]]; then common+=(--model_repo "${MODEL_REPO}"); fi
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT:-${TASK_DIR}/results}/megadna}"
cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/eval_megadna_promoter.py" "${common[@]}" --results_dir "${RESULTS_DIR}" --model_path "${MODEL_PATH:?Set MODEL_PATH to the local megaDNA checkpoint}")
if [[ -n "${FEATURE_CACHE_DIR:-}" ]]; then cmd+=(--feature_cache_dir "${FEATURE_CACHE_DIR}"); fi
exec "${cmd[@]}" "$@"
