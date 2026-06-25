#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/mldl/bin/python}

"$PYTHON" src/train.py --config configs/hybrid_no_transformer.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_depth1.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_depth4.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_patch2.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_patch8.json --device cuda "$@"
"$PYTHON" src/train.py --config configs/hybrid_wide.json --device cuda "$@"
