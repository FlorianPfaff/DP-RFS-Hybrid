import numpy as np
import pytest

from dp_rfs_hybrid import (
    MeasurementDrivenBirthModel,
    RECURRING_BIRTH_TRACKER_KINDS,
    GaussianState,
    make_recurring_birth_tracker,
    run_recurring_birth_trial,
    simulate_recurring_birth_scenario,
)


def test_recurring_birth_scenario_is_reproducible() -> None:
    first = simulate_recurring_birth_scenario(seed=4, scans=12)
    second = simulate_recurring_birth_scenario(seed=4, scans=12)

    assert np.allclose(first.target_states, second.target_states, equal_nan=True)
    assert all(
        np.allclose(first_scan, second_scan)
        for first_scan, second_scan in zip(first.measurements, second.measurements)
    )


def test_measurement_driven_birth_normalizes_batch_mass() -> None:
    model = MeasurementDrivenBirthModel(
        base_state=GaussianState(np.zeros(4), np.eye(4)),
        measurement_matrix=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        measurement_noise=np.eye(2),
        clutter_intensity=1e-3,
        expected_births=0.3,
        max_birth_probability=0.1,
    )

    model.begin_birth_batch(6)
    decision = model.process(np.array([1.0, 2.0]))

    assert decision.accepted
    assert decision.existence_probability == pytest.approx(0.05)


def test_all_recurring_birth_trackers_run_smoke_scenario() -> None:
    scenario = simulate_recurring_birth_scenario(seed=2, scans=6)

    for kind in RECURRING_BIRTH_TRACKER_KINDS:
        tracker = make_recurring_birth_tracker(kind)
        assert tracker is not None
        result = run_recurring_birth_trial(
            seed=2,
            tracker_kind=kind,
            scans=6,
            scenario=scenario,
        )
        assert np.isfinite(result.rms_gospa)
        assert result.scans == 6
