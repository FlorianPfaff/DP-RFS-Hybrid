from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

import numpy as np

from dp_rfs_hybrid.recurring_birth_benchmark import (
    RECURRING_BIRTH_TRACKER_KINDS,
    RecurringBirthMetrics,
    run_recurring_birth_trials,
)


def write_rows(rows: list[RecurringBirthMetrics], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(RecurringBirthMetrics)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[RecurringBirthMetrics]) -> None:
    for tracker in RECURRING_BIRTH_TRACKER_KINDS:
        group = [row for row in rows if row.tracker == tracker]
        if not group:
            continue
        gospa = np.asarray([row.rms_gospa for row in group])
        false = np.asarray([row.rms_false for row in group])
        atoms = np.asarray([row.final_birth_atoms for row in group])
        ci = 1.96 * np.std(gospa, ddof=1) / np.sqrt(len(gospa)) if len(gospa) > 1 else 0.0
        print(
            f"{tracker}: rms_gospa={np.mean(gospa):.3f} +/- {ci:.3f} "
            f"rms_false={np.mean(false):.3f} final_atoms={np.mean(atoms):.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100, help="number of paired seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument("--scans", type=int, default=96, help="scans per seed")
    parser.add_argument("--dp-alpha", type=float, default=10.0)
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=RECURRING_BIRTH_TRACKER_KINDS,
        default=list(RECURRING_BIRTH_TRACKER_KINDS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_metrics.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_recurring_birth_trials(
        seeds=range(args.seed_start, args.seed_start + args.seeds),
        scans=args.scans,
        tracker_kinds=tuple(args.trackers),
        dp_alpha=args.dp_alpha,
    )
    write_rows(rows, args.output)
    print_summary(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
