from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

from dp_rfs_hybrid.recurring_birth_benchmark import (
    RECURRING_BIRTH_REGIONS,
    RECURRING_BIRTH_VELOCITIES,
    SURVEILLANCE_BOUNDS,
    measurement_matrix,
    process_noise_covariance,
    simulate_recurring_birth_scenario,
    transition_matrix,
)


def export_dataset(seed_start: int, seeds: int, scans: int, output: Path) -> None:
    scenarios = [
        simulate_recurring_birth_scenario(seed=seed, scans=scans)
        for seed in range(seed_start, seed_start + seeds)
    ]
    target_count = max(len(scenario.target_birth_scans) for scenario in scenarios)
    max_measurements = max(
        len(measurements)
        for scenario in scenarios
        for measurements in scenario.measurements
    )
    target_states = np.full((4, target_count, scans, seeds), np.nan)
    target_alive = np.zeros((target_count, scans, seeds), dtype=np.uint8)
    measurement_values = np.full((2, max_measurements, scans, seeds), np.nan)
    measurement_counts = np.zeros((scans, seeds), dtype=np.int32)

    for seed_index, scenario in enumerate(scenarios):
        states = np.transpose(scenario.target_states, (2, 0, 1))
        target_states[:, : states.shape[1], :, seed_index] = states
        target_alive[: states.shape[1], :, seed_index] = np.isfinite(states[0])
        for scan, measurements in enumerate(scenario.measurements):
            measurement_count = len(measurements)
            measurement_counts[scan, seed_index] = measurement_count
            measurement_values[:, :measurement_count, scan, seed_index] = measurements.T

    base_mean = np.zeros((4, 1))
    base_covariance = np.diag([30.0**2, 22.0**2, 2.0**2, 2.0**2])
    oracle_means = np.vstack(
        (
            RECURRING_BIRTH_REGIONS.T,
            RECURRING_BIRTH_VELOCITIES.T,
        )
    )
    oracle_covariances = np.repeat(
        np.diag([1.2**2, 1.2**2, 0.35**2, 0.35**2])[:, :, None],
        len(RECURRING_BIRTH_REGIONS),
        axis=2,
    )
    area = float(np.prod(SURVEILLANCE_BOUNDS[:, 1] - SURVEILLANCE_BOUNDS[:, 0]))
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output,
        {
            "seed_values": np.arange(seed_start, seed_start + seeds, dtype=np.int32),
            "target_states": target_states,
            "target_alive": target_alive,
            "measurement_values": measurement_values,
            "measurement_counts": measurement_counts,
            "F": transition_matrix(),
            "Q": process_noise_covariance(),
            "H": measurement_matrix(),
            "R": np.diag([0.6**2, 0.6**2]),
            "p_d": 0.9,
            "p_s": 0.98,
            "clutter_rate": 6.0,
            "surveillance_area": area,
            "base_mean": base_mean,
            "base_covariance": base_covariance,
            "oracle_means": oracle_means,
            "oracle_covariances": oracle_covariances,
            "birth_rate": 0.15,
        },
        do_compression=True,
    )
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--scans", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recurring_birth_pmbm_dataset.mat"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dataset(args.seed_start, args.seeds, args.scans, args.output)


if __name__ == "__main__":
    main()
