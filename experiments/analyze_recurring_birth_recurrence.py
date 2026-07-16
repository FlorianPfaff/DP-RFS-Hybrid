from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analyze_recurring_birth_results import paired_comparison


REFERENCE = "dp_delayed_recluster"
COMPARATORS = ("fixed_broad_birth", "measurement_driven_birth")


def read_groups(path: Path) -> dict[int, dict[str, dict[int, float]]]:
    groups: dict[int, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[int(row["birth_events"])][row["tracker"]][int(row["seed"])] = float(
                row["rms_gospa"]
            )
    return {
        birth_events: dict(trackers)
        for birth_events, trackers in groups.items()
    }


def analyze(path: Path) -> list[dict[str, str | float | int]]:
    results: list[dict[str, str | float | int]] = []
    for birth_events, trackers in sorted(read_groups(path).items()):
        reference = trackers[REFERENCE]
        for comparator in COMPARATORS:
            comparison = paired_comparison(
                reference,
                trackers[comparator],
                random_seed=1729 + birth_events,
            )
            results.append(
                {
                    "birth_events": birth_events,
                    "reference": REFERENCE,
                    "comparator": comparator,
                    **comparison,
                }
            )
    return results


def write_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str | float | int]], path: Path) -> None:
    lines = [
        "# Recurring-birth recurrence comparisons",
        "",
        "Negative paired differences favor `dp_delayed_recluster`.",
        "",
        "| Births | Comparator | n | DP | Comparator | Difference (bootstrap 95% CI) | Improvement | p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['birth_events']} | {row['comparator']} | {row['n']} | "
            f"{float(row['reference_mean']):.3f} | "
            f"{float(row['comparator_mean']):.3f} | "
            f"{float(row['mean_paired_difference']):.3f} "
            f"[{float(row['bootstrap_ci95_low']):.3f}, "
            f"{float(row['bootstrap_ci95_high']):.3f}] | "
            f"{float(row['relative_improvement_percent']):.1f}% | "
            f"{float(row['sign_flip_p_value']):.3g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_recurrence_comparisons.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/recurring_birth_recurrence_comparisons.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze(args.input)
    write_csv(rows, args.output)
    write_markdown(rows, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
