#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
PhaGen paper training reference, using the VeOmni framework.

Usage (from the repository root):
  NPROC_PER_NODE=4 CONFIG=training/configs/paper_reference.yaml \
    bash training/train.sh [--model.model_path /path/to/initialization] \
    [--data.train_path /path/to/train.parquet] \
    [--data.val_path /path/to/validation.parquet] \
    [--train.output_dir /path/to/new/run] [other argument overrides]

Environment: CONFIG, NPROC_PER_NODE (default 4), NNODES, NODE_RANK,
MASTER_ADDR, MASTER_PORT. CUDA_VISIBLE_DEVICES is respected as supplied.
Relative CONFIG paths resolve from the calling directory; relative paths inside
the configuration resolve from training/. No old checkpoint is auto-resumed.
The original paper warm-start weights are not included; see training/README.md.
HELP
    exit 0
fi

CONFIG=${CONFIG:-"${TRAINING_ROOT}/configs/paper_reference.yaml"}
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${PWD}/${CONFIG}"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration does not exist: $CONFIG" >&2
    exit 2
fi

export PYTHONPATH="${TRAINING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
cd "$TRAINING_ROOT"

exec torchrun \
    --nnodes="${NNODES:-1}" \
    --nproc-per-node="${NPROC_PER_NODE:-4}" \
    --node-rank="${NODE_RANK:-0}" \
    --master-addr="${MASTER_ADDR:-127.0.0.1}" \
    --master-port="${MASTER_PORT:-29500}" \
    "${TRAINING_ROOT}/tasks/train_torch_dllm.py" "$CONFIG" "$@"

