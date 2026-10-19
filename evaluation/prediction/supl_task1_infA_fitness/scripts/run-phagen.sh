#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-${PHAGEN_MODEL:-YusenWithR/PhaGen}}"
REVISION="${REVISION:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-${TASK_DIR}/results/phagen}"
MUTANTS_PATH="${MUTANTS_PATH:-${TASK_DIR}/data/mutants_sequences.tsv}"
N_JOBS="${N_JOBS:-1}"
RUN_MODE="${RUN_MODE:-full}"
FORCE="${FORCE:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
REUSE_CACHED_FEATURES="${REUSE_CACHED_FEATURES:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/infA-phagen-infA_mutation.py"
  --model "${MODEL}"
  --results_dir "${RESULTS_DIR}"
  --mutants_path "${MUTANTS_PATH}"
  --device "${DEVICE}"
  --dtype float32
  --pad_attention_mode attend_pad
  --pool_pad_mode exclude_pad
  --add_bos_eos 0
  --strategies all_concat_mean
  --use_standard_scaler 1
  --n_jobs "${N_JOBS}"
  --run_mode "${RUN_MODE}"
  --progress_every "${PROGRESS_EVERY}"
)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  cmd+=(--local_files_only)
fi
if [[ -n "${REVISION}" ]]; then
  cmd+=(--revision "${REVISION}")
fi
if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi

if [[ "${REUSE_CACHED_FEATURES}" == "1" ]]; then
  cmd+=(--reuse_cached_features)
fi

exec "${cmd[@]}" "$@"
