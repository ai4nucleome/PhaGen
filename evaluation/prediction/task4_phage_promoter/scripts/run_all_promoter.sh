#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for family in phagen megadna evo evo2 ntv2 ntv3; do
  bash "${SCRIPT_DIR}/run_${family}_promoter.sh" "$@"
done
"${PYTHON_BIN:-python}" "${SCRIPT_DIR}/collect_results.py" --results_dir "${RESULTS_ROOT:-${SCRIPT_DIR}/../results}" --force
