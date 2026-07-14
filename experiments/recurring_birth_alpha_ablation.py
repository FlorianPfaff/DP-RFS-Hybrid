from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dp_rfs_hybrid.recurring_birth_benchmark import (
    run_recurring_birth_trial,
    simulate_recurring_birth_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--scans", type=int, default=96)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_alpha_ablation_seed100.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int]] = []
    for seed in range(args.seeds):
        scenario = simulate_recurring_birth_scenario(seed=seed, scans=args.scans)
        for alpha in args.alphas:
            result = run_recurring_birth_trial(
                seed=seed,
                tracker_kind="dp_delayed_recluster",
                scans=args.scans,
                scenario=scenario,
                dp_alpha=alpha,
            )
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "rms_gospa": result.rms_gospa,
                    "rms_missed": result.rms_missed,
                    "rms_false": result.rms_false,
                    "final_birth_atoms": result.final_birth_atoms,
                    "spurious_birth_atoms": result.spurious_birth_atoms,
                    "birth_region_error": result.birth_region_error,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    for alpha in args.alphas:
        values = np.asarray([row["rms_gospa"] for row in rows if row["alpha"] == alpha])
        ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        print(f"alpha={alpha:g}: rms_gospa={np.mean(values):.3f} +/- {ci:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
