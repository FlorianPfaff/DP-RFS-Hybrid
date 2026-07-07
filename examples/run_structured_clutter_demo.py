from __future__ import annotations

import numpy as np

from dp_rfs_hybrid import (
    DirichletProcessBirthModel,
    DirichletProcessClutterModel,
    GaussianState,
    LabeledMultiBernoulliTracker,
)


def make_tracker(use_adaptive_clutter: bool) -> LabeledMultiBernoulliTracker:
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
    if use_adaptive_clutter:
        clutter_model = DirichletProcessClutterModel(
            alpha=1.0,
            base_mean=np.array([10.0, 5.0]),
            base_covariance=np.diag([9.0, 9.0]),
            rate=8.0,
            prune_below_count=0.02,
            max_atoms=8,
        )
    return LabeledMultiBernoulliTracker(
        transition_matrix=transition,
        process_noise=process_noise,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        birth_model=birth_model,
        clutter_model=clutter_model,
        association_threshold=8.0,
        prune_below_existence=0.05,
    )


def simulate_measurements(rng: np.random.Generator, scan: int) -> np.ndarray:
    measurements: list[np.ndarray] = []

    # Persistent structured clutter near a background artifact.
    for _ in range(rng.poisson(3)):
        measurements.append(np.array([10.0, 5.0]) + rng.normal(scale=0.7, size=2))

    # Occasional true births away from the clutter hotspot.
    if scan in {0, 1, 2, 10, 11, 12}:
        measurements.append(np.array([-16.0, -5.0]) + rng.normal(scale=0.6, size=2))

    # Low-rate diffuse clutter.
    for _ in range(rng.poisson(1)):
        measurements.append(rng.uniform(low=[-30.0, -20.0], high=[30.0, 20.0]))

    rng.shuffle(measurements)
    return np.asarray(measurements, dtype=float)


def main() -> None:
    rng = np.random.default_rng(11)
    fixed = make_tracker(use_adaptive_clutter=False)
    adaptive = make_tracker(use_adaptive_clutter=True)

    print("scan measurements fixed_tracks adaptive_tracks clutter_atoms")
    for scan in range(20):
        measurements = simulate_measurements(rng, scan)
        fixed.step(measurements)
        adaptive.step(measurements)
        clutter_atoms = 0 if adaptive.clutter_model is None else len(adaptive.clutter_model.atoms)
        print(
            f"{scan:02d} {len(measurements):12d} "
            f"{len(fixed.estimates()):12d} {len(adaptive.estimates()):15d} {clutter_atoms:13d}"
        )

    if adaptive.clutter_model is not None:
        print("\nLearned clutter atoms:")
        for index, atom in enumerate(adaptive.clutter_model.atoms, start=1):
            print(
                f"  atom {index}: count={atom.count:.2f}, "
                f"mean=({atom.mean[0]:.2f}, {atom.mean[1]:.2f})"
            )


if __name__ == "__main__":
    main()
