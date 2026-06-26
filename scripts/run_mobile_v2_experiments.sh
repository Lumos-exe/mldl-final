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

"$PYTHON" src/train.py --config configs/mobile_cnn_matched.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_cifar_v2.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_v2_no_attention.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_v2_depth1.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/mobilevit_v2_patch4.json --device cuda "$@"
