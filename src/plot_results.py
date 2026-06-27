from __future__ import annotations

import argparse
import csv
import json
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

MAIN_EXPERIMENTS = [
    "compact_cnn",
    "compact_hybrid",
    "compact_hybrid_balanced",
]

ABLATION_EXPERIMENTS = [
    "compact_hybrid",
    "compact_no_attention",
    "compact_depth1",
    "compact_depth3",
    "compact_patch1",
    "compact_patch4",
]

EXPERIMENT_ORDER = MAIN_EXPERIMENTS + [
    "compact_no_attention",
    "compact_depth1",
    "compact_depth3",
    "compact_patch1",
    "compact_patch4",
]

LABELS = {
    "compact_cnn": "CNN",
    "compact_hybrid": "Hybrid",
    "compact_hybrid_balanced": "Hybrid-B",
    "compact_no_attention": "NoAttn",
    "compact_depth1": "Depth1",
    "compact_depth3": "Depth3",
    "compact_patch1": "Patch1",
    "compact_patch4": "Patch4",
}


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


def relative_deltas(values: list[float], baseline: float | None = None) -> list[float]:
    if not values:
        return []
    base = values[0] if baseline is None else baseline
    return [round(value - base, 2) for value in values]


def moving_average(values: list[float], window: int = 7) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return [float(value) for value in values]
    smoothed = []
    radius = window // 2
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        smoothed.append(round(sum(values[start:end]) / (end - start), 4))
    return smoothed


def padded_limits(values: list[float], pad: float = 0.18) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        return low - 1.0, high + 1.0
    return floor((low - pad) * 10) / 10, ceil((high + pad) * 10) / 10


def prepare_training_curve(
    rows: list[dict], min_epoch: int = 10, smooth_window: int = 3
) -> tuple[list[float], list[float]]:
    filtered = [
        (float(row["epoch"]), float(row["val_acc"]) * 100)
        for row in rows
        if float(row["epoch"]) >= min_epoch
    ]
    epochs = [epoch for epoch, _ in filtered]
    acc = [value for _, value in filtered]
    return epochs, moving_average(acc, window=smooth_window)


def set_style():
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", font_scale=0.88)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "font.family": "DejaVu Sans",
            "lines.solid_capstyle": "round",
        }
    )
    return {
        "cnn": "#4C78A8",
        "hybrid": "#F58518",
        "balanced": "#54A24B",
        "positive": "#2F9E44",
        "negative": "#D9480F",
        "neutral": "#8A8F98",
        "ablation": "#5AA9A6",
        "baseline": "#F58518",
        "line": ["#4C78A8", "#F58518", "#54A24B", "#D9480F", "#5AA9A6", "#9C6ADE"],
    }


def list_run_dirs(runs_dir: Path, wanted: list[str] | None):
    run_dirs = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir() and (p / "metrics.csv").exists() and (p / "summary.json").exists()
    ]
    if wanted:
        wanted_set = set(wanted)
        run_dirs = [p for p in run_dirs if p.name in wanted_set]
    order = {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(run_dirs, key=lambda p: (order.get(p.name, len(order)), p.name))


def select(run_dirs: list[Path], names: list[str]) -> list[Path]:
    by_name = {p.name: p for p in run_dirs}
    return [by_name[name] for name in names if name in by_name]


def pct(summary: dict, key: str) -> float:
    return float(summary[key]) * 100


def draw_vertical_accuracy(run_dirs: list[Path], out_dir: Path, filename: str, colors: list[str]):
    summaries = [read_summary(p) for p in run_dirs]
    labels = [LABELS[s["experiment"]] for s in summaries]
    accs = [pct(s, "test_acc") for s in summaries]
    deltas = relative_deltas(accs)

    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    ypos = list(range(len(labels)))
    bars = ax.barh(ypos, deltas, color=colors[: len(labels)], height=0.54)
    ax.axvline(0, color="#4D4D4D", linewidth=0.75)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy change vs CNN (pp)")
    ax.set_xlim(*padded_limits(deltas))
    ax.grid(axis="x", linewidth=0.45, alpha=0.42)
    ax.grid(axis="y", visible=False)
    for bar, delta, acc in zip(bars, deltas, accs):
        label_x = delta + (0.04 if delta >= 0 else -0.04)
        ha = "left" if delta >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.2f} ({acc:.2f}%)",
            va="center",
            ha=ha,
            fontsize=6.8,
        )
    savefig(fig, out_dir / filename)


def draw_ablation_accuracy(run_dirs: list[Path], out_dir: Path, colors: dict[str, str]):
    summaries = [read_summary(p) for p in run_dirs]
    labels = [LABELS[s["experiment"]] for s in summaries]
    accs = [pct(s, "test_acc") for s in summaries]
    baseline = accs[0]
    deltas = relative_deltas(accs, baseline=baseline)
    bar_colors = [
        colors["neutral"] if delta == 0 else colors["positive"] if delta > 0 else colors["negative"]
        for delta in deltas
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ypos = list(range(len(labels)))
    bars = ax.barh(ypos, deltas, color=bar_colors, height=0.54)
    ax.axvline(0, color="#4D4D4D", linewidth=0.75)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy change vs Hybrid (pp)")
    ax.set_xlim(*padded_limits(deltas, pad=0.22))
    ax.grid(axis="x", linewidth=0.45, alpha=0.45)
    ax.grid(axis="y", visible=False)
    for bar, delta, acc in zip(bars, deltas, accs):
        label = f"{delta:+.2f} ({acc:.2f}%)"
        if delta < -0.65:
            label_x = delta / 2
            ha = "center"
            color = "white"
        else:
            label_x = delta + (0.05 if delta >= 0 else -0.05)
            ha = "left" if delta >= 0 else "right"
            color = "#2B2B2B"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            color=color,
            fontsize=6.8,
        )
    savefig(fig, out_dir / "ablation_accuracy")


def draw_training_curves(run_dirs: list[Path], out_dir: Path, filename: str, colors: list[str], legend_cols: int = 1):
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    all_acc = []
    for i, run_dir in enumerate(run_dirs):
        rows = read_metrics(run_dir / "metrics.csv")
        epochs, acc = prepare_training_curve(rows)
        all_acc.extend(acc)
        ax.plot(
            epochs,
            acc,
            label=LABELS.get(run_dir.name, run_dir.name),
            linewidth=0.95,
            color=colors[i % len(colors)],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(floor((min(all_acc) - 2) / 5) * 5, ceil((max(all_acc) + 1) / 5) * 5)
    ax.grid(linewidth=0.45, alpha=0.45)
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor="#DDDDDD",
        ncol=legend_cols,
        loc="lower right",
        borderaxespad=0.35,
        handlelength=1.4,
        columnspacing=0.8,
    )
    savefig(fig, out_dir / filename)


def write_table(run_dirs: list[Path], out: Path):
    with out.open("w", encoding="utf-8") as f:
        for run_dir in run_dirs:
            s = read_summary(run_dir)
            f.write(
                f"{LABELS.get(s['experiment'], s['experiment'])} & "
                f"{s['parameters'] / 1e6:.3f} & "
                f"{s['best_val_acc'] * 100:.2f} & "
                f"{s['test_acc'] * 100:.2f} & "
                f"{s['best_epoch']} \\\\\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Create report figures from CIFAR-100 experiment logs.")
    parser.add_argument("--runs-dir", default="outputs/runs", type=Path)
    parser.add_argument("--out-dir", default="outputs/figures", type=Path)
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args()

    colors = set_style()
    run_dirs = list_run_dirs(args.runs_dir, args.experiments)
    if not run_dirs:
        raise SystemExit(f"No completed runs found in {args.runs_dir}")

    main_runs = select(run_dirs, MAIN_EXPERIMENTS)
    ablation_runs = select(run_dirs, ABLATION_EXPERIMENTS)
    if len(main_runs) != len(MAIN_EXPERIMENTS):
        missing = sorted(set(MAIN_EXPERIMENTS) - {p.name for p in main_runs})
        raise SystemExit(f"Missing main experiment runs: {missing}")
    if len(ablation_runs) != len(ABLATION_EXPERIMENTS):
        missing = sorted(set(ABLATION_EXPERIMENTS) - {p.name for p in ablation_runs})
        raise SystemExit(f"Missing ablation runs: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw_vertical_accuracy(
        main_runs,
        args.out_dir,
        "main_accuracy",
        [colors["cnn"], colors["hybrid"], colors["balanced"]],
    )
    draw_ablation_accuracy(ablation_runs, args.out_dir, colors)
    draw_training_curves(
        main_runs,
        args.out_dir,
        "main_training_curves",
        [colors["cnn"], colors["hybrid"], colors["balanced"]],
    )
    draw_training_curves(
        ablation_runs,
        args.out_dir,
        "ablation_training_curves",
        colors["line"],
        legend_cols=3,
    )
    write_table(main_runs, args.out_dir / "main_results_table.tex")
    write_table(ablation_runs, args.out_dir / "ablation_results_table.tex")
    print(f"Saved report figures and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
