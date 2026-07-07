from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dp_rfs_hybrid import (
    DirichletProcessBirthModel,
    DirichletProcessClutterModel,
    FixedGaussianMixtureClutterModel,
    GaussianState,
    LabeledMultiBernoulliTracker,
)


HOTSPOT = np.array([10.0, 5.0])
TRUE_BIRTH = np.array([-16.0, -5.0])
TRACKER_KINDS = (
    "fixed_scalar_clutter",
    "fixed_gmm_clutter",
    "adaptive_dp_clutter",
)


@dataclass(frozen=True)
class TrackerMetrics:
    seed: int
    tracker: str
    total_births: int
    final_estimated_tracks: int
    final_active_tracks: int
    hotspot_track_steps: int
    final_birth_atoms: int
    final_clutter_atoms: int


def make_tracker(kind: str) -> LabeledMultiBernoulliTracker:
    dt = 1.0
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    process_noise = np.diag([0.05, 0.05, 0.02, 0.02])
    measurement = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    measurement_noise = np.diag([0.4, 0.4])
    base_state = GaussianState(
        mean=np.array([0.0, 0.0, 0.0, 0.0]),
        covariance=np.diag([150.0, 150.0, 16.0, 16.0]),
    )
    birth_model = DirichletProcessBirthModel(
        alpha=5.0,
        base_state=base_state,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        clutter_intensity=1e-4,
        birth_probability=0.85,
        odds_threshold=20.0,
        max_atoms=12,
    )
    clutter_model = None
    if kind == "adaptive_dp_clutter":
        clutter_model = DirichletProcessClutterModel(
            alpha=1.0,
            base_mean=HOTSPOT.copy(),
            base_covariance=np.diag([9.0, 9.0]),
            rate=8.0,
            prune_below_count=0.02,
            max_atoms=8,
        )
    elif kind == "fixed_gmm_clutter":
        clutter_model = FixedGaussianMixtureClutterModel(
            weights=np.array([1.0]),
            means=np.array([HOTSPOT.copy()]),
            covariances=np.array([np.diag([1.5, 1.5])]),
            rate=8.0,
        )
    elif kind != "fixed_scalar_clutter":
        raise ValueError(f"Unknown tracker kind: {kind}")

    return LabeledMultiBernoulliTracker(
        transition_matrix=transition,
        process_noise=process_noise,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        birth_model=birth_model,
        clutter_model=clutter_model,
        association_threshold=8.0,
        prune_below_existence=0.05,
        min_clutter_responsibility_to_learn=0.05,
    )


def simulate_measurements(rng: np.random.Generator, scan: int) -> np.ndarray:
    measurements: list[np.ndarray] = []

    for _ in range(rng.poisson(3)):
        measurements.append(HOTSPOT + rng.normal(scale=0.7, size=2))

    if scan in {0, 1, 2, 10, 11, 12}:
        measurements.append(TRUE_BIRTH + rng.normal(scale=0.6, size=2))

    for _ in range(rng.poisson(1)):
        measurements.append(rng.uniform(low=[-30.0, -20.0], high=[30.0, 20.0]))

    rng.shuffle(measurements)
    return np.asarray(measurements, dtype=float)


def count_hotspot_estimates(tracker: LabeledMultiBernoulliTracker, radius: float = 3.0) -> int:
    count = 0
    for track in tracker.estimates():
        if np.linalg.norm(track.state.mean[:2] - HOTSPOT) <= radius:
            count += 1
    return count


def count_clutter_atoms(tracker: LabeledMultiBernoulliTracker) -> int:
    clutter_model = tracker.clutter_model
    if clutter_model is None or not hasattr(clutter_model, "atoms"):
        return 0
    return len(clutter_model.atoms)


def run_one(seed: int, tracker_kind: str, scans: int) -> TrackerMetrics:
    rng = np.random.default_rng(seed)
    tracker = make_tracker(tracker_kind)
    total_births = 0
    hotspot_track_steps = 0

    for scan in range(scans):
        summary = tracker.step(simulate_measurements(rng, scan))
        total_births += len(summary.births)
        hotspot_track_steps += count_hotspot_estimates(tracker)

    return TrackerMetrics(
        seed=seed,
        tracker=tracker_kind,
        total_births=total_births,
        final_estimated_tracks=len(tracker.estimates()),
        final_active_tracks=len(tracker.tracks),
        hotspot_track_steps=hotspot_track_steps,
        final_birth_atoms=len(tracker.birth_model.atoms),
        final_clutter_atoms=count_clutter_atoms(tracker),
    )


def run_many(seeds: range, scans: int) -> list[TrackerMetrics]:
    rows: list[TrackerMetrics] = []
    for seed in seeds:
        for tracker_kind in TRACKER_KINDS:
            rows.append(run_one(seed, tracker_kind, scans))
    return rows


def write_csv(rows: list[TrackerMetrics], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TrackerMetrics.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[TrackerMetrics]) -> None:
    trackers = sorted({row.tracker for row in rows})
    for tracker in trackers:
        group = [row for row in rows if row.tracker == tracker]
        mean_births = np.mean([row.total_births for row in group])
        mean_hotspot_steps = np.mean([row.hotspot_track_steps for row in group])
        mean_clutter_atoms = np.mean([row.final_clutter_atoms for row in group])
        print(
            f"{tracker}: mean_births={mean_births:.2f} "
            f"mean_hotspot_track_steps={mean_hotspot_steps:.2f} "
            f"mean_final_clutter_atoms={mean_clutter_atoms:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="number of random seeds")
    parser.add_argument("--scans", type=int, default=20, help="number of scans per run")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/structured_clutter_metrics.csv"),
        help="CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_many(range(args.seeds), args.scans)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
