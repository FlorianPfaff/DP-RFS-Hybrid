from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "fixed_broad_birth": "Fixed broad",
    "measurement_driven_birth": "MDB",
    "dp_immediate": "DP immediate",
    "dp_delayed_append": "DP delayed, append",
    "dp_delayed_recluster": "DP delayed, recluster",
    "oracle_birth": "Oracle birth",
    "pmbm_broad": "PMBM broad",
    "pmbm_oracle": "PMBM oracle",
}
ORDER = tuple(DISPLAY_NAMES)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def filter_seed_range(
    rows: list[dict[str, str]],
    seed_start: int | None,
    seed_stop: int | None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (seed_start is None or int(row["seed"]) >= seed_start)
        and (seed_stop is None or int(row["seed"]) < seed_stop)
    ]


def grouped_values(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, np.ndarray]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            groups[row["tracker"]].append(value)
    return {key: np.asarray(value) for key, value in groups.items()}


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def available_order(groups: dict[str, np.ndarray]) -> list[str]:
    return [tracker for tracker in ORDER if tracker in groups]


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_gospa(rows: list[dict[str, str]], output_dir: Path) -> Path:
    groups = grouped_values(rows, "rms_gospa")
    trackers = available_order(groups)
    means, cis = zip(*(mean_ci(groups[tracker]) for tracker in trackers))
    positions = np.arange(len(trackers))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#6b7280"] * len(trackers)
    if "dp_delayed_recluster" in trackers:
        colors[trackers.index("dp_delayed_recluster")] = "#0072b2"
    if "oracle_birth" in trackers:
        colors[trackers.index("oracle_birth")] = "#009e73"
    if "pmbm_oracle" in trackers:
        colors[trackers.index("pmbm_oracle")] = "#009e73"
    ax.bar(positions, means, yerr=cis, capsize=3, color=colors, width=0.72)
    for position, mean in zip(positions, means):
        ax.text(position, mean + max(means) * 0.025, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, [DISPLAY_NAMES[tracker] for tracker in trackers], rotation=24, ha="right")
    ax.set_ylabel("RMS GOSPA (m)")
    _style_axis(ax)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "recurring_birth_gospa.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_decomposition(rows: list[dict[str, str]], output_dir: Path) -> Path:
    metrics = ("rms_localization", "rms_missed", "rms_false")
    labels = ("Localization", "Missed", "False")
    primary_groups = grouped_values(rows, metrics[0])
    trackers = available_order(primary_groups)
    positions = np.arange(len(trackers))
    width = 0.24
    colors = ("#0072b2", "#e69f00", "#d55e00")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for metric_index, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        groups = grouped_values(rows, metric)
        means = [mean_ci(groups[tracker])[0] for tracker in trackers]
        cis = [mean_ci(groups[tracker])[1] for tracker in trackers]
        offset = (metric_index - 1) * width
        ax.bar(positions + offset, means, width=width, yerr=cis, capsize=2, label=label, color=color)
    ax.set_xticks(positions, [DISPLAY_NAMES[tracker] for tracker in trackers], rotation=24, ha="right")
    ax.set_ylabel("RMS GOSPA component (m)")
    ax.legend(frameon=False, ncol=3)
    _style_axis(ax)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "recurring_birth_gospa_decomposition.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_learning_effect(rows: list[dict[str, str]], output_dir: Path) -> Path:
    early = grouped_values(rows, "early_rms_gospa")
    late = grouped_values(rows, "late_rms_gospa")
    trackers = [tracker for tracker in available_order(early) if tracker in late]
    positions = np.arange(len(trackers))
    width = 0.36
    early_means = [mean_ci(early[tracker])[0] for tracker in trackers]
    late_means = [mean_ci(late[tracker])[0] for tracker in trackers]
    early_cis = [mean_ci(early[tracker])[1] for tracker in trackers]
    late_cis = [mean_ci(late[tracker])[1] for tracker in trackers]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(positions - width / 2, early_means, width, yerr=early_cis, capsize=2, label="Scans 1-32", color="#9ca3af")
    ax.bar(positions + width / 2, late_means, width, yerr=late_cis, capsize=2, label="Scans 33-96", color="#0072b2")
    ax.set_xticks(positions, [DISPLAY_NAMES[tracker] for tracker in trackers], rotation=24, ha="right")
    ax.set_ylabel("RMS GOSPA (m)")
    ax.legend(frameon=False, ncol=2)
    _style_axis(ax)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "recurring_birth_learning_effect.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    metrics = (
        "rms_gospa",
        "rms_localization",
        "rms_missed",
        "rms_false",
        "mean_cardinality_error",
        "final_birth_atoms",
        "spurious_birth_atoms",
        "birth_region_error",
        "runtime_seconds",
    )
    trackers = available_order(grouped_values(rows, "rms_gospa"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["tracker", "n"] + [part for metric in metrics for part in (f"{metric}_mean", f"{metric}_ci95")]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for tracker in trackers:
            output_row: dict[str, str | int | float] = {"tracker": tracker}
            n = 0
            for metric in metrics:
                groups = grouped_values(rows, metric)
                values = groups.get(tracker, np.asarray([]))
                n = max(n, len(values))
                if len(values):
                    mean, ci = mean_ci(values)
                else:
                    mean, ci = float("nan"), float("nan")
                output_row[f"{metric}_mean"] = mean
                output_row[f"{metric}_ci95"] = ci
            output_row["n"] = n
            writer.writerow(output_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--summary", type=Path, default=Path("results/recurring_birth_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_seed_range(
        read_rows(args.input),
        seed_start=args.seed_start,
        seed_stop=args.seed_stop,
    )
    if not rows:
        raise ValueError("no rows remain after seed filtering")
    write_summary(rows, args.summary)
    for output in (
        plot_gospa(rows, args.output_dir),
        plot_decomposition(rows, args.output_dir),
        plot_learning_effect(rows, args.output_dir),
    ):
        print(f"wrote {output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
