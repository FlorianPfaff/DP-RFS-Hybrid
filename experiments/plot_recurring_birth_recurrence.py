from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "fixed_broad_birth": ("Fixed broad", "#6b7280", "s"),
    "measurement_driven_birth": ("MDB", "#e69f00", "^"),
    "dp_delayed_recluster": ("DP delayed, recluster", "#0072b2", "o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/recurring_birth_recurrence.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["tracker"], int(row["birth_events"]))
            groups[key].append(float(row["rms_gospa"]))

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for tracker, (label, color, marker) in DISPLAY.items():
        birth_events = sorted(
            events for candidate, events in groups if candidate == tracker
        )
        means = np.asarray([np.mean(groups[(tracker, events)]) for events in birth_events])
        cis = np.asarray(
            [
                1.96
                * np.std(groups[(tracker, events)], ddof=1)
                / np.sqrt(len(groups[(tracker, events)]))
                for events in birth_events
            ]
        )
        ax.errorbar(
            birth_events,
            means,
            yerr=cis,
            label=label,
            color=color,
            marker=marker,
            capsize=3,
        )

    ax.set_xticks(sorted({events for _, events in groups}))
    ax.set_xlabel("Birth events per 96 scans")
    ax.set_ylabel("RMS GOSPA (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
