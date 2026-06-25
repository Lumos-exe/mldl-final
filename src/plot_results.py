from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:  # The existing course container has matplotlib; seaborn is optional.
    sns = None


def read_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in list(row):
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    return rows


def read_summary(run_dir: Path):
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def set_style():
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
        return sns.color_palette("Set2")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    return ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]


def main():
    parser = argparse.ArgumentParser(description="Create publication-style figures from experiment logs.")
    parser.add_argument("--runs-dir", default="outputs/runs", type=Path)
    parser.add_argument("--out-dir", default="outputs/figures", type=Path)
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args()

    palette = set_style()
    run_dirs = [
        p
        for p in args.runs_dir.iterdir()
        if p.is_dir() and (p / "metrics.csv").exists() and (p / "summary.json").exists()
    ]
    if args.experiments:
        wanted = set(args.experiments)
        run_dirs = [p for p in run_dirs if p.name in wanted]
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise SystemExit(f"No completed runs found in {args.runs_dir}")

    summaries = [read_summary(p) for p in run_dirs]
    labels = [s["experiment"] for s in summaries]
    accs = [s["best_test_acc"] * 100 for s in summaries]
    params = [s["parameters"] / 1e6 for s in summaries]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, accs, color=palette[: len(labels)])
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, max(accs) * 1.18 if accs else 100)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    savefig(fig, args.out_dir / "accuracy_comparison")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, params, color=palette[: len(labels)])
    ax.set_ylabel("Trainable parameters (M)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(params) * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    savefig(fig, args.out_dir / "parameter_comparison")

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, label in enumerate(labels):
        ax.scatter(params[i], accs[i], s=80, color=palette[i % len(palette)], label=label)
        ax.annotate(label, (params[i], accs[i]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Trainable parameters (M)")
    ax.set_ylabel("Best test accuracy (%)")
    savefig(fig, args.out_dir / "accuracy_params_tradeoff")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, run_dir in enumerate(run_dirs):
        rows = read_metrics(run_dir / "metrics.csv")
        epochs = [r["epoch"] for r in rows]
        test_acc = [r["test_acc"] * 100 for r in rows]
        ax.plot(epochs, test_acc, label=run_dir.name, linewidth=2, color=palette[i % len(palette)])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=True, fontsize=8)
    savefig(fig, args.out_dir / "training_curves")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "results_table.tex").open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(f"{s['experiment']} & {s['parameters'] / 1e6:.2f} & {s['best_test_acc'] * 100:.2f} & {s['best_epoch']} \\\\\n")
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
