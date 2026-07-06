import numpy as np

from dp_rfs_hybrid import DirichletProcessBirthModel, GaussianState, LabeledMultiBernoulliTracker


def make_tracker() -> LabeledMultiBernoulliTracker:
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
    return LabeledMultiBernoulliTracker(
        transition_matrix=transition,
        process_noise=process_noise,
        measurement_matrix=measurement,
        measurement_noise=measurement_noise,
        birth_model=birth_model,
        association_threshold=5.0,
        prune_below_existence=0.01,
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
