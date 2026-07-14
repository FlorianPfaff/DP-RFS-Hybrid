from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REFERENCE = "dp_delayed_recluster"
COMPARATORS = (
    "fixed_broad_birth",
    "measurement_driven_birth",
    "dp_immediate",
    "dp_delayed_append",
    "oracle_birth",
    "pmbm_broad",
    "pmbm_oracle",
)


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


def index_metric(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, dict[int, float]]:
    indexed: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            indexed[row["tracker"]][int(row["seed"])] = value
    return dict(indexed)


def paired_comparison(
    reference: dict[int, float],
    comparator: dict[int, float],
    random_seed: int = 1729,
) -> dict[str, float | int]:
    seeds = sorted(set(reference) & set(comparator))
    reference_values = np.asarray([reference[seed] for seed in seeds])
    comparator_values = np.asarray([comparator[seed] for seed in seeds])
    differences = reference_values - comparator_values
    count = len(differences)
    mean_difference = float(np.mean(differences))
    if count > 1:
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(count))
        ci95 = 1.96 * standard_error
        z_score = abs(mean_difference / standard_error) if standard_error > 0.0 else math.inf
        p_value = math.erfc(z_score / math.sqrt(2.0))
    else:
        ci95 = float("nan")
        p_value = float("nan")
    rng = np.random.default_rng(random_seed)
    if count > 1:
        bootstrap_indices = rng.integers(0, count, size=(20_000, count))
        bootstrap_means = np.mean(differences[bootstrap_indices], axis=1)
        bootstrap_low, bootstrap_high = np.quantile(bootstrap_means, [0.025, 0.975])
        sign_flips = rng.choice(np.array([-1.0, 1.0]), size=(50_000, count))
        null_means = np.mean(sign_flips * differences, axis=1)
        sign_flip_p_value = (
            1.0 + np.sum(np.abs(null_means) >= abs(mean_difference))
        ) / (len(null_means) + 1.0)
    else:
        bootstrap_low = float("nan")
        bootstrap_high = float("nan")
        sign_flip_p_value = float("nan")
    comparator_mean = float(np.mean(comparator_values))
    relative_improvement = (
        100.0 * (comparator_mean - float(np.mean(reference_values))) / comparator_mean
        if comparator_mean != 0.0
        else float("nan")
    )
    return {
        "n": count,
        "reference_mean": float(np.mean(reference_values)),
        "comparator_mean": comparator_mean,
        "mean_paired_difference": mean_difference,
        "paired_difference_ci95": ci95,
        "bootstrap_ci95_low": float(bootstrap_low),
        "bootstrap_ci95_high": float(bootstrap_high),
        "relative_improvement_percent": relative_improvement,
        "normal_approx_p_value": p_value,
        "sign_flip_p_value": float(sign_flip_p_value),
    }


def write_comparisons(
    rows: list[dict[str, str]],
    metric: str,
    output: Path,
) -> list[dict[str, str | float | int]]:
    indexed = index_metric(rows, metric)
    comparisons: list[dict[str, str | float | int]] = []
    reference = indexed[REFERENCE]
    for comparator_name in COMPARATORS:
        comparator = indexed.get(comparator_name)
        if not comparator:
            continue
        result = paired_comparison(reference, comparator)
        comparisons.append(
            {
                "metric": metric,
                "reference": REFERENCE,
                "comparator": comparator_name,
                **result,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(comparisons[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons


def write_markdown(
    comparisons: list[dict[str, str | float | int]],
    output: Path,
) -> None:
    lines = [
        "# Recurring-birth paired comparisons",
        "",
        "Negative paired differences favor `dp_delayed_recluster`.",
        "",
        "| Comparator | n | DP | Comparator | Difference (bootstrap 95% CI) | Improvement | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        difference = float(row["mean_paired_difference"])
        ci_low = float(row["bootstrap_ci95_low"])
        ci_high = float(row["bootstrap_ci95_high"])
        lines.append(
            f"| {row['comparator']} | {row['n']} | "
            f"{float(row['reference_mean']):.3f} | "
            f"{float(row['comparator_mean']):.3f} | "
            f"{difference:.3f} [{ci_low:.3f}, {ci_high:.3f}] | "
            f"{float(row['relative_improvement_percent']):.1f}% | "
            f"{float(row['sign_flip_p_value']):.3g} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", default="rms_gospa")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_paired_comparisons.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/recurring_birth_paired_comparisons.md"),
    )
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
    comparisons = write_comparisons(rows, args.metric, args.output)
    write_markdown(comparisons, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
