"""Recurring-birth benchmark for delayed DP birth learning."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .birth_baselines import (
    FixedGaussianMixtureBirthModel,
    MeasurementDrivenBirthModel,
)
from .dp_birth import DirichletProcessBirthModel
from .gaussian import GaussianState
from .lmb_tracker import LabeledMultiBernoulliTracker
from .metrics import gospa


SURVEILLANCE_BOUNDS = np.array([[-50.0, 50.0], [-35.0, 35.0]])
RECURRING_BIRTH_REGIONS = np.array([[-24.0, -10.0], [22.0, 12.0]])
RECURRING_BIRTH_VELOCITIES = np.array([[1.15, 0.35], [-1.0, -0.3]])
RECURRING_BIRTH_TRACKER_KINDS = (
    "fixed_broad_birth",
    "measurement_driven_birth",
    "dp_immediate",
    "dp_delayed_append",
    "dp_delayed_recluster",
    "oracle_birth",
)


@dataclass(frozen=True)
class RecurringBirthScenario:
    """One simulated truth and measurement realization."""

    target_states: np.ndarray
    target_birth_scans: np.ndarray
    target_death_scans: np.ndarray
    target_region_indices: np.ndarray
    measurements: tuple[np.ndarray, ...]

    @property
    def scans(self) -> int:
        return int(self.target_states.shape[1])

    def truth_positions(self, scan: int) -> np.ndarray:
        states = self.target_states[:, scan]
        alive = np.isfinite(states[:, 0])
        return states[alive, :2]


@dataclass(frozen=True)
class RecurringBirthMetrics:
    """Aggregate metrics for one seed and one tracker configuration."""

    seed: int
    tracker: str
    scans: int
    total_births: int
    total_confirmed_births: int
    rms_gospa: float
    rms_localization: float
    rms_missed: float
    rms_false: float
    early_rms_gospa: float
    late_rms_gospa: float
    mean_cardinality_error: float
    mean_false_targets: float
    mean_missed_targets: float
    final_birth_atoms: int
    recurrent_birth_atoms: int
    spurious_birth_atoms: int
    birth_region_error: float
    runtime_seconds: float


def transition_matrix(dt: float = 1.0) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def process_noise_covariance(acceleration_std: float = 0.06, dt: float = 1.0) -> np.ndarray:
    gain = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return acceleration_std**2 * gain @ gain.T + 1e-6 * np.eye(4)


def measurement_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )


def simulate_recurring_birth_scenario(
    seed: int,
    scans: int = 96,
    spawn_interval: int = 8,
    target_lifetime: int = 14,
    detection_probability: float = 0.9,
    clutter_rate: float = 6.0,
    measurement_std: float = 0.6,
) -> RecurringBirthScenario:
    """Simulate targets repeatedly emerging from two unknown regions."""

    if scans <= 0:
        raise ValueError("scans must be positive")
    if spawn_interval <= 0 or target_lifetime <= 0:
        raise ValueError("spawn_interval and target_lifetime must be positive")
    if not 0.0 <= detection_probability <= 1.0:
        raise ValueError("detection_probability must be in [0, 1]")
    if clutter_rate < 0.0 or measurement_std <= 0.0:
        raise ValueError("clutter_rate must be nonnegative and measurement_std positive")

    rng = np.random.default_rng(seed)
    birth_scans = np.arange(0, scans, spawn_interval, dtype=int)
    target_count = len(birth_scans)
    death_scans = np.minimum(birth_scans + target_lifetime, scans)
    region_indices = np.arange(target_count, dtype=int) % len(RECURRING_BIRTH_REGIONS)
    target_states = np.full((target_count, scans, 4), np.nan)
    transition = transition_matrix()
    acceleration_gain = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    for target_index, birth_scan in enumerate(birth_scans):
        region_index = region_indices[target_index]
        state = np.concatenate(
            (
                RECURRING_BIRTH_REGIONS[region_index]
                + rng.normal(scale=0.55, size=2),
                RECURRING_BIRTH_VELOCITIES[region_index]
                + rng.normal(scale=0.08, size=2),
            )
        )
        for scan in range(birth_scan, death_scans[target_index]):
            target_states[target_index, scan] = state
            acceleration = rng.normal(scale=0.06, size=2)
            state = transition @ state + acceleration_gain @ acceleration

    measurements: list[np.ndarray] = []
    low = SURVEILLANCE_BOUNDS[:, 0]
    high = SURVEILLANCE_BOUNDS[:, 1]
    for scan in range(scans):
        scan_measurements: list[np.ndarray] = []
        truth = target_states[:, scan]
        for state in truth[np.isfinite(truth[:, 0])]:
            if rng.random() <= detection_probability:
                scan_measurements.append(
                    state[:2] + rng.normal(scale=measurement_std, size=2)
                )
        for _ in range(rng.poisson(clutter_rate)):
            scan_measurements.append(rng.uniform(low=low, high=high))
        rng.shuffle(scan_measurements)
        measurements.append(np.asarray(scan_measurements, dtype=float).reshape(-1, 2))

    return RecurringBirthScenario(
        target_states=target_states,
        target_birth_scans=birth_scans,
        target_death_scans=death_scans,
        target_region_indices=region_indices,
        measurements=tuple(measurements),
    )


def _base_state() -> GaussianState:
    return GaussianState(
        mean=np.zeros(4),
        covariance=np.diag([30.0**2, 22.0**2, 2.0**2, 2.0**2]),
    )


def _oracle_states() -> list[GaussianState]:
    return [
        GaussianState(
            mean=np.concatenate((region, velocity)),
            covariance=np.diag([1.2**2, 1.2**2, 0.35**2, 0.35**2]),
        )
        for region, velocity in zip(
            RECURRING_BIRTH_REGIONS,
            RECURRING_BIRTH_VELOCITIES,
        )
    ]


def make_recurring_birth_tracker(
    kind: str,
    dp_alpha: float = 10.0,
) -> LabeledMultiBernoulliTracker:
    """Create one controlled birth-model configuration."""

    if kind not in RECURRING_BIRTH_TRACKER_KINDS:
        raise ValueError(f"Unknown recurring-birth tracker kind: {kind}")
    if dp_alpha <= 0.0:
        raise ValueError("dp_alpha must be positive")
    measurement = measurement_matrix()
    measurement_noise = np.diag([0.6**2, 0.6**2])
    surveillance_area = float(np.prod(SURVEILLANCE_BOUNDS[:, 1] - SURVEILLANCE_BOUNDS[:, 0]))
    clutter_intensity = 6.0 / surveillance_area
    base_state = _base_state()
    delayed = False

    if kind == "fixed_broad_birth":
        birth_model = FixedGaussianMixtureBirthModel(
            weights=np.array([1.0]),
            states=[base_state],
            measurement_matrix=measurement,
            measurement_noise=measurement_noise,
            clutter_intensity=clutter_intensity,
            birth_rate=0.15,
            birth_probability=0.35,
            odds_threshold=0.02,
        )
    elif kind == "measurement_driven_birth":
        birth_model = MeasurementDrivenBirthModel(
            base_state=base_state,
            measurement_matrix=measurement,
            measurement_noise=measurement_noise,
            clutter_intensity=clutter_intensity,
            expected_births=0.15,
            max_birth_probability=0.15,
        )
    elif kind == "oracle_birth":
        birth_model = FixedGaussianMixtureBirthModel(
            weights=np.array([0.5, 0.5]),
            states=_oracle_states(),
            measurement_matrix=measurement,
            measurement_noise=measurement_noise,
            clutter_intensity=clutter_intensity,
            birth_rate=0.15,
            birth_probability=0.35,
            odds_threshold=0.02,
        )
    else:
        delayed = kind != "dp_immediate"
        birth_model = DirichletProcessBirthModel(
            alpha=dp_alpha,
            base_state=base_state,
            measurement_matrix=measurement,
            measurement_noise=measurement_noise,
            clutter_intensity=clutter_intensity,
            birth_probability=0.35,
            birth_rate=0.15,
            odds_threshold=0.02,
            max_atoms=24,
            prune_below_count=0.05,
            recluster_confirmed_states=kind != "dp_delayed_append",
        )

    return LabeledMultiBernoulliTracker(
        transition_matrix=transition_matrix(),
        process_noise=process_noise_covariance(),
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        birth_model=birth_model,
        survival_probability=0.98,
        detection_probability=0.9,
        association_threshold=0.2,
        prune_below_existence=0.01,
        max_tracks=128,
        delayed_birth_learning=delayed,
        birth_confirmation_age=1,
        birth_confirmation_existence=0.6,
    )


def _root_mean_powered(values: list[float]) -> float:
    return float(np.sqrt(np.mean(values))) if values else 0.0


def _birth_atom_diagnostics(
    tracker: LabeledMultiBernoulliTracker,
    kind: str,
) -> tuple[int, int, int, float]:
    if kind == "oracle_birth":
        return 2, 2, 0, 0.0
    if not isinstance(tracker.birth_model, DirichletProcessBirthModel):
        return 0, 0, 0, float("nan")

    atoms = tracker.birth_model.atoms
    recurrent_atoms = [atom for atom in atoms if atom.count >= 1.5]
    if not atoms:
        return 0, 0, 0, float("inf")
    atom_positions = np.asarray([atom.state.mean[:2] for atom in atoms])
    region_distances = np.linalg.norm(
        RECURRING_BIRTH_REGIONS[:, None, :] - atom_positions[None, :, :],
        axis=2,
    )
    region_error = float(np.mean(np.min(region_distances, axis=1)))
    spurious_atoms = sum(
        np.min(np.linalg.norm(RECURRING_BIRTH_REGIONS - atom.state.mean[:2], axis=1)) > 4.0
        for atom in recurrent_atoms
    )
    return len(atoms), len(recurrent_atoms), int(spurious_atoms), region_error


def run_recurring_birth_trial(
    seed: int,
    tracker_kind: str,
    scans: int = 96,
    scenario: RecurringBirthScenario | None = None,
    dp_alpha: float = 10.0,
) -> RecurringBirthMetrics:
    """Run one tracker on one shared recurring-birth realization."""

    if scenario is None:
        scenario = simulate_recurring_birth_scenario(seed=seed, scans=scans)
    if scenario.scans != scans:
        raise ValueError("scenario scan count does not match scans")
    tracker = make_recurring_birth_tracker(tracker_kind, dp_alpha=dp_alpha)
    total_births = 0
    total_confirmed_births = 0
    powered_gospa: list[float] = []
    localization: list[float] = []
    missed: list[float] = []
    false: list[float] = []
    cardinality_errors: list[float] = []
    false_counts: list[float] = []
    missed_counts: list[float] = []
    started = time.perf_counter()

    for scan, measurements in enumerate(scenario.measurements):
        summary = tracker.step(measurements)
        total_births += len(summary.births)
        total_confirmed_births += len(summary.confirmed_births)
        estimates = np.asarray(
            [track.state.mean[:2] for track in tracker.estimates()],
            dtype=float,
        ).reshape(-1, 2)
        truth = scenario.truth_positions(scan)
        result = gospa(truth, estimates, cutoff=10.0, order=2.0, alpha=2.0)
        powered_gospa.append(result.powered_distance)
        localization.append(result.localization)
        missed.append(result.missed)
        false.append(result.false)
        cardinality_errors.append(abs(len(estimates) - len(truth)))
        false_counts.append(result.false_count)
        missed_counts.append(result.missed_count)

    runtime = time.perf_counter() - started
    split = min(32, scans)
    final_atoms, recurrent_atoms, spurious_atoms, region_error = _birth_atom_diagnostics(
        tracker,
        tracker_kind,
    )
    return RecurringBirthMetrics(
        seed=seed,
        tracker=tracker_kind,
        scans=scans,
        total_births=total_births,
        total_confirmed_births=total_confirmed_births,
        rms_gospa=_root_mean_powered(powered_gospa),
        rms_localization=_root_mean_powered(localization),
        rms_missed=_root_mean_powered(missed),
        rms_false=_root_mean_powered(false),
        early_rms_gospa=_root_mean_powered(powered_gospa[:split]),
        late_rms_gospa=_root_mean_powered(powered_gospa[split:]),
        mean_cardinality_error=float(np.mean(cardinality_errors)),
        mean_false_targets=float(np.mean(false_counts)),
        mean_missed_targets=float(np.mean(missed_counts)),
        final_birth_atoms=final_atoms,
        recurrent_birth_atoms=recurrent_atoms,
        spurious_birth_atoms=spurious_atoms,
        birth_region_error=region_error,
        runtime_seconds=float(runtime),
    )


def run_recurring_birth_trials(
    seeds: range,
    scans: int = 96,
    tracker_kinds: tuple[str, ...] = RECURRING_BIRTH_TRACKER_KINDS,
    dp_alpha: float = 10.0,
) -> list[RecurringBirthMetrics]:
    """Run all tracker kinds on paired random seeds."""

    rows: list[RecurringBirthMetrics] = []
    for seed in seeds:
        scenario = simulate_recurring_birth_scenario(seed=seed, scans=scans)
        for tracker_kind in tracker_kinds:
            rows.append(
                run_recurring_birth_trial(
                    seed=seed,
                    tracker_kind=tracker_kind,
                    scans=scans,
                    scenario=scenario,
                    dp_alpha=dp_alpha,
                )
            )
    return rows
