#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/mldl/bin/python}

"$PYTHON" src/train.py --config configs/cnn_baseline.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/tiny_vit.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_main.json --device cuda "$@"
