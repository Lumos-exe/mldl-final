#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif [[ -x /root/miniconda3/envs/mldl/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/mldl/bin/python
  else
    echo "Python not found. Set PYTHON=/path/to/python before running this script." >&2
    exit 1
  fi
fi
BASE=configs/compact_hybrid.json
COMMON=(--config "$BASE" --device cuda)

"$PYTHON" src/train.py "${COMMON[@]}" --experiment compact_no_attention --mixer no_attention "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment compact_depth1 --depth 1 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment compact_depth3 --depth 3 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment compact_patch1 --patch-size 1 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment compact_patch4 --patch-size 4 "$@"
