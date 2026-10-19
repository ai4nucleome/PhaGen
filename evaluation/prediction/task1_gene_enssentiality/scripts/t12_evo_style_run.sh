#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PY_PHAGEN="${PY_PHAGEN:-${PYTHON_BIN}}"
PY_MEGADNA="${PY_MEGADNA:-${PYTHON_BIN}}"
PY_EVO1="${PY_EVO1:-${PYTHON_BIN}}"
PY_EVO2="${PY_EVO2:-${PYTHON_BIN}}"
PY_NTV3="${PY_NTV3:-${PYTHON_BIN}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
models=(phagen megadna evo1 evo2_1b evo2 NTv3_8M_pre NTv3_100M_pre NTv3_650M_pre)
common=(--phage lambda --context 8192 --device "${DEVICE:-cuda:0}")
dry_run=0

usage() {
    cat <<EOF
Usage: bash $0 [OPTIONS]

Run the paper's eight models with the insert/8192-bp/12-nt-offset/15-nt-cassette protocol.

  --models M [M ...]     phagen megadna evo1 evo2_1b evo2 NTv3_8M_pre NTv3_100M_pre NTv3_650M_pre
  --phage NAME           lambda or p1 (default: lambda)
  --device DEVICE        Inference device (default: DEVICE or cuda:0)
  --data-dir DIR         Task input directory
  --results-dir DIR      Task result directory
  --max-genes N          Small-run limit; 0 means all labelled CDSs
  --force                Replace existing result files
  --dry-run              Print commands without running them
  --help                 Show this help

Use PYTHON_BIN or PY_PHAGEN/PY_MEGADNA/PY_EVO1/PY_EVO2/PY_NTV3 for model environments.
Set PHAGEN_MODEL, MEGADNA_MODEL_PATH and optional baseline repository variables for local weights.
EVO2_1B_MODEL_PATH/EVO2_7B_MODEL_PATH and NTV3_8M_MODEL_PATH/NTV3_100M_MODEL_PATH/NTV3_650M_MODEL_PATH select local checkpoints.
Model downloads are disabled unless ALLOW_DOWNLOAD=1.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            models=()
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                models+=("$1")
                shift
            done
            [[ ${#models[@]} -gt 0 ]] || { echo "--models requires at least one model" >&2; exit 2; }
            ;;
        --phage|--context|--mutation-mode|--mutation-offset-nt|--cassette-nt|--gff-type|--device|--data-dir|--results-dir|--max-genes)
            common+=("$1" "${2:?Missing option value}")
            shift 2
            ;;
        --force) common+=(--force); shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for model in "${models[@]}"; do
    local_path=""
    case "${model}" in
        phagen)
            cmd=("${PY_PHAGEN}" "${SCRIPT_DIR}/t12-evo-style-phagen.py" --model "${PHAGEN_MODEL:-YusenWithR/PhaGen}")
            ;;
        megadna)
            cmd=("${PY_MEGADNA}" "${SCRIPT_DIR}/t12-evo-style-megadna.py")
            ;;
        evo1)
            cmd=("${PY_EVO1}" "${SCRIPT_DIR}/t12-evo-style-evo1.py")
            ;;
        evo2_1b)
            cmd=("${PY_EVO2}" "${SCRIPT_DIR}/t12-evo-style-evo2.py" --model-name evo2_1b_base)
            local_path="${EVO2_1B_MODEL_PATH:-}"
            ;;
        evo2|evo2_7b)
            cmd=("${PY_EVO2}" "${SCRIPT_DIR}/t12-evo-style-evo2.py" --model-name evo2_7b)
            local_path="${EVO2_7B_MODEL_PATH:-}"
            ;;
        NTv3_8M_pre)
            cmd=("${PY_NTV3}" "${SCRIPT_DIR}/t12-evo-style-ntv3.py" --model-name "${model}")
            local_path="${NTV3_8M_MODEL_PATH:-}"
            ;;
        NTv3_100M_pre)
            cmd=("${PY_NTV3}" "${SCRIPT_DIR}/t12-evo-style-ntv3.py" --model-name "${model}")
            local_path="${NTV3_100M_MODEL_PATH:-}"
            ;;
        NTv3_650M_pre)
            cmd=("${PY_NTV3}" "${SCRIPT_DIR}/t12-evo-style-ntv3.py" --model-name "${model}")
            local_path="${NTV3_650M_MODEL_PATH:-}"
            ;;
        *) echo "Unknown paper model: ${model}" >&2; exit 2 ;;
    esac
    if [[ -n "${local_path}" ]]; then
        cmd+=(--model-path "${local_path}")
    fi
    if [[ "${ALLOW_DOWNLOAD}" == "1" && "${model}" != "megadna" ]]; then
        cmd+=(--allow-download)
    fi
    cmd+=("${common[@]}")
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" != "1" ]]; then
        "${cmd[@]}"
    fi
done
