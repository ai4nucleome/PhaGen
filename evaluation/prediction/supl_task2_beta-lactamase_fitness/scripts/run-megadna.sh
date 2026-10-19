#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_DIR}/lactamase-megadna-beta_lactamase.py"

MODEL_PATH="${MODEL_PATH:-${MEGADNA_MODEL_PATH:-}}"
MODEL_REPO="${MODEL_REPO:-${MEGADNA_REPO:-}}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-${TASK_DIR}/results/megadna}"
MUTANTS_PATH="${MUTANTS_PATH:-${TASK_DIR}/data/processed/mutants_sequences.tsv}"
N_JOBS="${N_JOBS:-1}"
RUN_MODE="${RUN_MODE:-full}"
REUSE_CACHED_FEATURES="${REUSE_CACHED_FEATURES:-0}"
FORCE="${FORCE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_PATH}"
  --results_dir "${RESULTS_DIR}"
  --mutants_path "${MUTANTS_PATH}"
  --device "${DEVICE}"
  --dtype float32
  --strategies all_concat_mean
  --add_bos_eos 0
  --use_standard_scaler 1
  --n_jobs "${N_JOBS}"
  --run_mode "${RUN_MODE}"
)

if [[ -n "${MODEL_REPO}" ]]; then
  cmd+=(--model_repo "${MODEL_REPO}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model_path "${MODEL_PATH}")
fi

if [[ "${REUSE_CACHED_FEATURES}" == "1" ]]; then
  cmd+=(--reuse_cached_features)
fi

if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi

exec "${cmd[@]}" "$@"
