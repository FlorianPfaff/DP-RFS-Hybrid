import numpy as np
import pytest

from dp_rfs_hybrid import (
    DirichletProcessBirthModel,
    DirichletProcessClutterModel,
    GaussianState,
    LabeledMultiBernoulliTracker,
)


def make_tracker(
    clutter_model: DirichletProcessClutterModel | None = None,
    **tracker_overrides,
) -> LabeledMultiBernoulliTracker:
    transition = np.eye(4)
    process_noise = np.diag([0.05, 0.05, 0.01, 0.01])
    measurement = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    measurement_noise = np.diag([0.25, 0.25])
    birth_model = DirichletProcessBirthModel(
        alpha=6.0,
        base_state=GaussianState(np.zeros(4), np.diag([100.0, 100.0, 25.0, 25.0])),
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        clutter_intensity=1e-4,
        birth_probability=0.8,
        odds_threshold=10.0,
    )
    kwargs = {
        "transition_matrix": transition,
        "process_noise": process_noise,
        "measurement_matrix": measurement,
        "measurement_noise": measurement_noise,
        "birth_model": birth_model,
        "clutter_model": clutter_model,
        "association_threshold": 5.0,
        "prune_below_existence": 0.01,
    }
    kwargs.update(tracker_overrides)
    return LabeledMultiBernoulliTracker(**kwargs)


def make_hotspot_clutter_model() -> DirichletProcessClutterModel:
    return DirichletProcessClutterModel(
        alpha=1.0,
        base_mean=np.array([8.0, 2.0]),
        base_covariance=np.diag([0.25, 0.25]),
        rate=10.0,
        prune_below_count=0.01,
    )


def test_unassigned_measurement_spawns_track() -> None:
    tracker = make_tracker()

    summary = tracker.step(np.array([[8.0, 2.0]]))

    assert summary.births == [1]
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].label == 1


def test_nearby_followup_measurement_updates_existing_track() -> None:
    tracker = make_tracker()
    tracker.step(np.array([[8.0, 2.0]]))

    summary = tracker.step(np.array([[8.1, 2.1]]))

    assert summary.assignments == [(1, 0)]
    assert summary.births == []
    assert len(tracker.tracks) == 1


def test_dp_clutter_model_can_suppress_birth_and_learn_hotspot() -> None:
    clutter_model = make_hotspot_clutter_model()
    tracker = make_tracker(clutter_model=clutter_model)

    summary = tracker.step(np.array([[8.0, 2.0]]))

    assert summary.births == []
    assert summary.clutter == [0]
    assert len(tracker.tracks) == 0
    assert len(clutter_model.atoms) == 1
    assert summary.clutter_updates[0][0] == 0
    assert summary.clutter_updates[0][1] > 0.99


def test_clutter_responsibility_learning_rate_attenuates_updates() -> None:
    clutter_model = make_hotspot_clutter_model()
    tracker = make_tracker(
        clutter_model=clutter_model,
        clutter_responsibility_learning_rate=0.25,
    )

    summary = tracker.step(np.array([[8.0, 2.0]]))

    assert len(clutter_model.atoms) == 1
    assert summary.clutter_updates[0][1] == pytest.approx(0.25, abs=1e-3)
    # The tracker applies clutter-model count decay at the end of each scan.
    assert clutter_model.atoms[0].count == pytest.approx(
        summary.clutter_updates[0][1] * 0.995,
        rel=1e-6,
    )


def test_min_clutter_responsibility_gate_can_skip_learning() -> None:
    clutter_model = make_hotspot_clutter_model()
    tracker = make_tracker(
        clutter_model=clutter_model,
        min_clutter_responsibility_to_learn=1.0,
    )

    summary = tracker.step(np.array([[8.0, 2.0]]))

    assert summary.clutter == [0]
    assert len(clutter_model.atoms) == 0
    assert summary.clutter_updates[0][1] == 0.0


def test_delayed_birth_learning_defers_birth_atom_update_until_confirmation() -> None:
    tracker = make_tracker(
        delayed_birth_learning=True,
        birth_confirmation_age=1,
        birth_confirmation_existence=0.5,
    )

    first_summary = tracker.step(np.array([[8.0, 2.0]]))

    assert first_summary.births == [1]
    assert first_summary.confirmed_births == []
    assert len(tracker.birth_model.atoms) == 0
    assert tracker.tracks[0].pending_birth_learning

    second_summary = tracker.step(np.array([[8.1, 2.1]]))

    assert second_summary.assignments == [(1, 0)]
    assert second_summary.confirmed_births == [1]
    assert len(tracker.birth_model.atoms) == 1
    assert not tracker.tracks[0].pending_birth_learning


def test_delayed_birth_learning_respects_confirmation_threshold() -> None:
    tracker = make_tracker(
        delayed_birth_learning=True,
        birth_confirmation_age=1,
        birth_confirmation_existence=1.0,
    )

    tracker.step(np.array([[8.0, 2.0]]))
    second_summary = tracker.step(np.array([[8.1, 2.1]]))

    assert second_summary.confirmed_births == []
    assert len(tracker.birth_model.atoms) == 0
    assert tracker.tracks[0].pending_birth_learning


def test_delayed_birth_learning_validates_confirmation_parameters() -> None:
    with pytest.raises(ValueError, match="birth_confirmation_age"):
        make_tracker(delayed_birth_learning=True, birth_confirmation_age=-1)

    with pytest.raises(ValueError, match="birth_confirmation_existence"):
        make_tracker(delayed_birth_learning=True, birth_confirmation_existence=1.5)


def test_clutter_learning_configuration_validates_probabilities() -> None:
    with pytest.raises(ValueError, match="clutter_responsibility_learning_rate"):
        make_tracker(clutter_responsibility_learning_rate=1.5)

    with pytest.raises(ValueError, match="min_clutter_responsibility_to_learn"):
        make_tracker(min_clutter_responsibility_to_learn=-0.1)
