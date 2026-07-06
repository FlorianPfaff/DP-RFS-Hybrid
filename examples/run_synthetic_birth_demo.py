from __future__ import annotations

import numpy as np

from dp_rfs_hybrid import DirichletProcessBirthModel, GaussianState, LabeledMultiBernoulliTracker


def make_tracker() -> LabeledMultiBernoulliTracker:
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
        covariance=np.diag([100.0, 100.0, 16.0, 16.0]),
    )
    birth_model = DirichletProcessBirthModel(
        alpha=4.0,
        base_state=base_state,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        clutter_intensity=1e-4,
        birth_probability=0.85,
        odds_threshold=20.0,
        max_atoms=8,
    )
    return LabeledMultiBernoulliTracker(
        transition_matrix=transition,
        process_noise=process_noise,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        birth_model=birth_model,
        association_threshold=8.0,
        prune_below_existence=0.05,
    )


def simulate_measurements(rng: np.random.Generator, scan: int) -> np.ndarray:
    centers = [np.array([-18.0, -4.0]), np.array([16.0, 7.0])]
    measurements: list[np.ndarray] = []
    for center in centers:
        if scan in {0, 1, 2, 8, 9, 15, 16}:
            measurements.append(center + rng.normal(scale=0.7, size=2))
    clutter_count = rng.poisson(2)
    for _ in range(clutter_count):
        measurements.append(rng.uniform(low=[-30.0, -20.0], high=[30.0, 20.0]))
    rng.shuffle(measurements)
    return np.asarray(measurements, dtype=float)


def main() -> None:
    rng = np.random.default_rng(7)
    tracker = make_tracker()
    for scan in range(20):
        measurements = simulate_measurements(rng, scan)
        summary = tracker.step(measurements)
        estimates = tracker.estimates()
        print(
            f"scan={scan:02d} measurements={len(measurements):2d} "
            f"births={len(summary.births):1d} tracks={len(estimates):2d} "
            f"atoms={len(tracker.birth_model.atoms):1d}"
        )

    print("\nLearned birth atoms:")
    for index, atom in enumerate(tracker.birth_model.atoms, start=1):
        position = atom.state.mean[:2]
        print(f"  atom {index}: count={atom.count:.2f}, position=({position[0]:.2f}, {position[1]:.2f})")


if __name__ == "__main__":
    main()
