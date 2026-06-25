from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_cifar100_loaders
from models import build_model, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch_idx, (images, targets) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), targets) * batch_size
        total_samples += batch_size
        iterator.set_postfix(loss=total_loss / total_samples, acc=total_acc / total_samples)

    return total_loss / max(total_samples, 1), total_acc / max(total_samples, 1)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight CNN/Transformer models on CIFAR-100.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default="/workspaces/mldl/CIFAR-100/data")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get("experiment", args.config.stem)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)

    train_cfg = config.get("training", {})
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 2))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    params = count_parameters(model)

    train_loader, test_loader = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=bool(train_cfg.get("augment", True)),
        randaugment=bool(train_cfg.get("randaugment", True)),
        download=args.download,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.1)))
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-2)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-6)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = Path(args.output_dir) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.json")
    metrics_path = run_dir / "metrics.csv"
    best_acc = -1.0
    best_epoch = 0
    start_time = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_seconds"],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=args.limit_train_batches,
            )
            test_loss, test_acc = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                max_batches=args.limit_val_batches,
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch_seconds": time.time() - epoch_start,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{exp_name}] epoch {epoch:03d}/{epochs} "
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} test_loss={test_loss:.4f}"
            )
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "config": config, "epoch": epoch, "test_acc": test_acc},
                    run_dir / "best.pt",
                )

        torch.save(
            {"model": model.state_dict(), "config": config, "epoch": epochs, "test_acc": test_acc},
            run_dir / "last.pt",
        )

    summary = {
        "experiment": exp_name,
        "config": str(args.config),
        "model": config.get("model", {}).get("name"),
        "parameters": params,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "epochs": epochs,
        "total_seconds": time.time() - start_time,
        "seed": seed,
        "device": str(device),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
