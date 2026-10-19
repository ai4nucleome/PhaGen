#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/t12_evo_style_run.sh" \
    --models NTv3_8M_pre NTv3_100M_pre NTv3_650M_pre "$@"
