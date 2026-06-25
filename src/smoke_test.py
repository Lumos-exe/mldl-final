from __future__ import annotations

import json
from pathlib import Path

import torch

from models import build_model, count_parameters


def main():
    configs = sorted(Path("configs").glob("*.json"))
    x = torch.randn(4, 3, 32, 32)
    for cfg_path in configs:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model = build_model(cfg)
        y = model(x)
        assert y.shape == (4, 100), f"{cfg_path}: got {tuple(y.shape)}"
        print(f"{cfg_path.name:30s} output={tuple(y.shape)} params={count_parameters(model):,}")


if __name__ == "__main__":
    main()
