from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from train import build_model, count_parameters


def main():
    configs = sorted(Path("configs").glob("*.json"))
    x = torch.randn(4, 3, 32, 32)
    for cfg_path in configs:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model = build_model(cfg)
        y = model(x)
        assert y.shape == (4, 100), f"{cfg_path}: got {tuple(y.shape)}"
        print(f"{cfg_path.name:30s} output={tuple(y.shape)} params={count_parameters(model):,}")

    hybrid_cfg = json.loads(Path("configs/hybrid_main.json").read_text(encoding="utf-8"))
    ablations = {
        "hybrid_no_attention": {"mixer": "no_attention"},
        "hybrid_depth2": {"depth": 2},
        "hybrid_depth6": {"depth": 6},
        "hybrid_patch1": {"patch_size": 1},
        "hybrid_patch4": {"patch_size": 4},
    }
    for name, overrides in ablations.items():
        cfg = copy.deepcopy(hybrid_cfg)
        cfg["model"].update(overrides)
        model = build_model(cfg)
        y = model(x)
        assert y.shape == (4, 100), f"{name}: got {tuple(y.shape)}"
        print(f"{name:30s} output={tuple(y.shape)} params={count_parameters(model):,}")


if __name__ == "__main__":
    main()
