#!/usr/bin/env bash
set -euo pipefail

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
BASE=configs/hybrid_main.json
COMMON=(--config "$BASE" --device cuda)

"$PYTHON" src/train.py "${COMMON[@]}" --experiment hybrid_no_attention --mixer no_attention "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment hybrid_depth1 --depth 1 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment hybrid_depth4 --depth 4 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment hybrid_patch2 --patch-size 2 "$@"
"$PYTHON" src/train.py "${COMMON[@]}" --experiment hybrid_patch8 --patch-size 8 "$@"
