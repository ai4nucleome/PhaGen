#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENCODER="${TASK_DIR}/../task5_pvp/scripts/encode.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_KEY="${1:-}"
if [[ -z "${MODEL_KEY}" || "${MODEL_KEY}" == "--help" || "${MODEL_KEY}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash scripts/run_embedding_model.sh MODEL_KEY [ENCODER_OPTIONS...]

Models: phagen megadna evo evo2_1b evo2_7b ntv2_50m ntv2_100m
        ntv2_250m ntv2_500m ntv3_8m ntv3_100m ntv3_650m
Environment: PYTHON_BIN, DATA_PATH, CACHE_ROOT, DEVICE, FORCE.
Extra options are passed to the shared pure encoder in task5_pvp/scripts/encode.py.
Prepare an embedding table with prepare_data.py --embedding-output PATH first.
EOF
    exit 0
fi
shift
DATA_PATH="${DATA_PATH:-${TASK_DIR}/data/embedding_input.tsv}"
CACHE_ROOT="${CACHE_ROOT:-${TASK_DIR}/feature_cache}"
cmd=("${PYTHON_BIN}" "${ENCODER}" --model-key "${MODEL_KEY}"
     --data-path "${DATA_PATH}" --sequence-column dna_seq
     --cache-dir "${CACHE_ROOT}/${MODEL_KEY}" --device "${DEVICE:-cuda:0}"
     --pool-special-tokens exclude)
if [[ "${FORCE:-0}" == "1" ]]; then cmd+=(--force); fi
"${cmd[@]}" "$@"
