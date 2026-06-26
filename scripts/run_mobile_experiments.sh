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

"$PYTHON" src/train.py --config configs/mobile_cnn.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_cifar.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_no_attention.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_depth1.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_patch4.json --device cuda "$@"
