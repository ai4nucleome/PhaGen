#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_DIR}/lactamase-ntv3-beta_lactamase.py"

MODELS="${MODELS:-NTv3_8M_pre NTv3_100M_pre NTv3_650M_pre}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_ROOT="${RESULTS_ROOT:-${TASK_DIR}/results}"
MUTANTS_PATH="${MUTANTS_PATH:-${TASK_DIR}/data/processed/mutants_sequences.tsv}"
N_JOBS="${N_JOBS:-1}"
RUN_MODE="${RUN_MODE:-full}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
REUSE_CACHED_FEATURES="${REUSE_CACHED_FEATURES:-0}"
FORCE="${FORCE:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
PAD_TO_MULTIPLE_OF="${PAD_TO_MULTIPLE_OF:-128}"
ADD_SPECIAL_TOKENS="${ADD_SPECIAL_TOKENS:-0}"
POOL_SPECIAL_TOKENS="${POOL_SPECIAL_TOKENS:-include}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="$([[ "${ALLOW_DOWNLOAD}" == "1" ]] && echo 0 || echo 1)"

for MODEL in ${MODELS}; do
  MODEL_BASENAME="${MODEL##*/}"
  MODEL_KEY="$(echo "${MODEL_BASENAME}" | tr '[:upper:]' '[:lower:]' | sed 's/_pre$//')"
  RESULTS_DIR="${RESULTS_ROOT}/${MODEL_KEY}"

  cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_PATH}"
    --results_dir "${RESULTS_DIR}"
    --mutants_path "${MUTANTS_PATH}"
    --model_name "${MODEL}"
    --device "${DEVICE}"
    --torch_dtype "${TORCH_DTYPE}"
    --pad_to_multiple_of "${PAD_TO_MULTIPLE_OF}"
    --add_special_tokens "${ADD_SPECIAL_TOKENS}"
    --pool_special_tokens "${POOL_SPECIAL_TOKENS}"
    --use_standard_scaler 1
    --n_jobs "${N_JOBS}"
    --run_mode "${RUN_MODE}"
    --progress_every "${PROGRESS_EVERY}"
  )

  if [[ "${REUSE_CACHED_FEATURES}" == "1" ]]; then
    cmd+=(--reuse_cached_features)
  fi
  if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
    cmd+=(--allow_download)
  fi

  if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
  fi

  echo "Running ${MODEL} -> ${RESULTS_DIR}"
  "${cmd[@]}" "$@"
done
