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

"$PYTHON" src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
if command -v latexmk >/dev/null 2>&1; then
  cd report
  latexmk -xelatex main.tex
else
  echo "latexmk not found; figures were generated, but the LaTeX report was not compiled." >&2
fi
