from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "hotspot_track_steps",
    "total_births",
    "final_estimated_tracks",
    "final_clutter_atoms",
)


def read_metric_groups(csv_path: Path, metric: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            groups[row["tracker"]].append(float(row[metric]))
    return dict(groups)


def plot_metric(csv_path: Path, metric: str, output_dir: Path) -> Path:
    groups = read_metric_groups(csv_path, metric)
    trackers = sorted(groups)
    values = [groups[tracker] for tracker in trackers]
    means = [float(np.mean(v)) for v in values]
    stds = [float(np.std(v)) for v in values]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(trackers, means, yerr=stds, capsize=4)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("tracker")
    ax.set_title(metric.replace("_", " ").title())
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"structured_clutter_{metric}.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/structured_clutter_metrics.csv"),
        help="CSV file produced by structured_clutter_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="directory for generated PDF figures",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(METRICS),
        help="metric columns to plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for metric in args.metrics:
        output_path = plot_metric(args.input, metric, args.output_dir)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
