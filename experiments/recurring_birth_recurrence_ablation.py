from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dp_rfs_hybrid.recurring_birth_benchmark import (
    run_recurring_birth_trial,
    simulate_recurring_birth_scenario,
)


TRACKERS = (
    "fixed_broad_birth",
    "measurement_driven_birth",
    "dp_delayed_recluster",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=80)
    parser.add_argument("--scans", type=int, default=96)
    parser.add_argument(
        "--spawn-intervals",
        type=int,
        nargs="+",
        default=[6, 8, 12, 16],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_recurrence_ablation_seed80.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for spawn_interval in args.spawn_intervals:
            scenario = simulate_recurring_birth_scenario(
                seed=seed,
                scans=args.scans,
                spawn_interval=spawn_interval,
            )
            for tracker in TRACKERS:
                result = run_recurring_birth_trial(
                    seed=seed,
                    tracker_kind=tracker,
                    scans=args.scans,
                    scenario=scenario,
                )
                rows.append(
                    {
                        "seed": seed,
                        "spawn_interval": spawn_interval,
                        "birth_events": len(scenario.target_birth_scans),
                        "tracker": tracker,
                        "rms_gospa": result.rms_gospa,
                        "rms_missed": result.rms_missed,
                        "rms_false": result.rms_false,
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

    for spawn_interval in args.spawn_intervals:
        for tracker in TRACKERS:
            values = np.asarray(
                [
                    float(row["rms_gospa"])
                    for row in rows
                    if row["spawn_interval"] == spawn_interval
                    and row["tracker"] == tracker
                ]
            )
            print(
                f"interval={spawn_interval} tracker={tracker}: "
                f"rms_gospa={np.mean(values):.3f}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
