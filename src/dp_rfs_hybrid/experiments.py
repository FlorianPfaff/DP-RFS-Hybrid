"""Reusable synthetic experiments for DP/RFS hybrid prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dp_birth import DirichletProcessBirthModel
from .dp_clutter import DirichletProcessClutterModel
from .gaussian import GaussianState
from .lmb_tracker import LabeledMultiBernoulliTracker


@dataclass(frozen=True)
class StructuredClutterScanRecord:
    """Per-scan metrics for the structured-clutter comparison."""

    scan: int
    measurement_count: int
    fixed_birth_count: int
    adaptive_birth_count: int
    fixed_active_track_count: int
    adaptive_active_track_count: int
    adaptive_clutter_atom_count: int


@dataclass(frozen=True)
class StructuredClutterExperimentResult:
    """Summary of a fixed-clutter versus adaptive-DP-clutter run."""

    records: tuple[StructuredClutterScanRecord, ...]
    fixed_total_births: int
    adaptive_total_births: int
    fixed_final_track_count: int
    adaptive_final_track_count: int
    adaptive_final_clutter_atom_count: int
    adaptive_clutter_atom_means: tuple[tuple[float, float], ...]

    def as_rows(self) -> list[dict[str, float | int]]:
        """Return per-scan records as dictionaries for tables or plotting."""

        return [
            {
                "scan": record.scan,
                "measurement_count": record.measurement_count,
                "fixed_birth_count": record.fixed_birth_count,
                "adaptive_birth_count": record.adaptive_birth_count,
                "fixed_active_track_count": record.fixed_active_track_count,
                "adaptive_active_track_count": record.adaptive_active_track_count,
                "adaptive_clutter_atom_count": record.adaptive_clutter_atom_count,
            }
            for record in self.records
        ]


def make_structured_clutter_tracker(use_adaptive_clutter: bool) -> LabeledMultiBernoulliTracker:
    """Create the tracker used by the structured-clutter experiment."""

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


def simulate_structured_clutter_measurements(
    rng: np.random.Generator,
    scan: int,
) -> np.ndarray:
    """Simulate measurements with one persistent clutter hotspot and occasional births."""

    measurements: list[np.ndarray] = []

    hotspot_count = max(1, int(rng.poisson(3)))
    for _ in range(hotspot_count):
        measurements.append(np.array([10.0, 5.0]) + rng.normal(scale=0.7, size=2))

    if scan in {0, 1, 2, 10, 11, 12}:
        measurements.append(np.array([-16.0, -5.0]) + rng.normal(scale=0.6, size=2))

    for _ in range(int(rng.poisson(1))):
        measurements.append(rng.uniform(low=[-30.0, -20.0], high=[30.0, 20.0]))

    rng.shuffle(measurements)
    return np.asarray(measurements, dtype=float)


def run_structured_clutter_experiment(
    scans: int = 20,
    seed: int = 11,
) -> StructuredClutterExperimentResult:
    """Run a fixed-clutter versus adaptive-DP-clutter comparison."""

    rng = np.random.default_rng(seed)
    fixed = make_structured_clutter_tracker(use_adaptive_clutter=False)
    adaptive = make_structured_clutter_tracker(use_adaptive_clutter=True)
    records: list[StructuredClutterScanRecord] = []
    fixed_total_births = 0
    adaptive_total_births = 0

    for scan in range(scans):
        measurements = simulate_structured_clutter_measurements(rng, scan)
        fixed_summary = fixed.step(measurements)
        adaptive_summary = adaptive.step(measurements)
        fixed_total_births += len(fixed_summary.births)
        adaptive_total_births += len(adaptive_summary.births)
        adaptive_clutter_atom_count = 0
        if adaptive.clutter_model is not None:
            adaptive_clutter_atom_count = len(adaptive.clutter_model.atoms)
        records.append(
            StructuredClutterScanRecord(
                scan=scan,
                measurement_count=int(len(measurements)),
                fixed_birth_count=len(fixed_summary.births),
                adaptive_birth_count=len(adaptive_summary.births),
                fixed_active_track_count=len(fixed.estimates()),
                adaptive_active_track_count=len(adaptive.estimates()),
                adaptive_clutter_atom_count=adaptive_clutter_atom_count,
            )
        )

    adaptive_clutter_atom_means: tuple[tuple[float, float], ...] = ()
    adaptive_final_clutter_atom_count = 0
    if adaptive.clutter_model is not None:
        adaptive_final_clutter_atom_count = len(adaptive.clutter_model.atoms)
        adaptive_clutter_atom_means = tuple(
            (float(atom.mean[0]), float(atom.mean[1]))
            for atom in adaptive.clutter_model.atoms
        )

    return StructuredClutterExperimentResult(
        records=tuple(records),
        fixed_total_births=fixed_total_births,
        adaptive_total_births=adaptive_total_births,
        fixed_final_track_count=len(fixed.estimates()),
        adaptive_final_track_count=len(adaptive.estimates()),
        adaptive_final_clutter_atom_count=adaptive_final_clutter_atom_count,
        adaptive_clutter_atom_means=adaptive_clutter_atom_means,
    )
