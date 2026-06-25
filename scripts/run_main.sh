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

"$PYTHON" src/train.py --config configs/cnn_baseline.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/tiny_vit.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_main.json --device cuda "$@"
