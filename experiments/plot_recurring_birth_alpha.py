from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/recurring_birth_alpha_sensitivity.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[float, list[float]] = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if args.seed_start is not None and seed < args.seed_start:
                continue
            if args.seed_stop is not None and seed >= args.seed_stop:
                continue
            groups[float(row["alpha"])].append(float(row["rms_gospa"]))
    if not groups:
        raise ValueError("no rows remain after seed filtering")
    alphas = np.asarray(sorted(groups))
    means = np.asarray([np.mean(groups[alpha]) for alpha in alphas])
    cis = np.asarray(
        [
            1.96 * np.std(groups[alpha], ddof=1) / np.sqrt(len(groups[alpha]))
            for alpha in alphas
        ]
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.errorbar(alphas, means, yerr=cis, marker="o", color="#0072b2", capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(alphas, [f"{alpha:g}" for alpha in alphas])
    ax.set_xlabel("DP concentration alpha")
    ax.set_ylabel("RMS GOSPA (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
