#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_DIR}/lactamase-evo2-beta_lactamase.py"

MODEL_PATH="${MODEL_PATH:-${EVO2_MODEL_PATH:-}}"
MODEL_REPO="${MODEL_REPO:-${EVO2_REPO:-}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-${TASK_DIR}/results/evo2_1b}"
MUTANTS_PATH="${MUTANTS_PATH:-${TASK_DIR}/data/processed/mutants_sequences.tsv}"
MODEL_NAME="${MODEL_NAME:-evo2_1b_base}"
PRIMARY_LAYER="${PRIMARY_LAYER:-blocks.15}"
N_JOBS="${N_JOBS:-1}"
RUN_MODE="${RUN_MODE:-full}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
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
  --model_name "${MODEL_NAME}"
  --device "${DEVICE}"
  --primary_layer "${PRIMARY_LAYER}"
  --strategies mean
  --use_standard_scaler 1
  --n_jobs "${N_JOBS}"
  --run_mode "${RUN_MODE}"
  --progress_every "${PROGRESS_EVERY}"
)

if [[ -n "${MODEL_REPO}" ]]; then
  cmd+=(--model_repo "${MODEL_REPO}")
fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
  cmd+=(--allow_download)
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
