#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/mldl/bin/python}

"$PYTHON" src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
cd report
latexmk -xelatex main.tex
