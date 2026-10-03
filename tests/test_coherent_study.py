import numpy as np
import pytest

from dp_rfs_hybrid.coherent_evidence import BirthLikelihood
from dp_rfs_hybrid.coherent_mixture import exact_posterior
from dp_rfs_hybrid.coherent_study import (
    BirthAdapter, HistoryTracker, Sensor, configurations, replay_stream, scenario,
)


def test_nearby_reuse_and_distant_creation():
    def site(label, x):
        return BirthLikelihood(label, (0,), [[[10.]]], [[10*x]], [-5*x*x])
    args = (np.zeros(1), np.eye(1)*100, np.eye(1)*.1)
    nearby = exact_posterior([site(1, 0), site(2, .1)], *args)
    distant = exact_posterior([site(1, 0), site(2, 8)], *args)
    assert nearby["coassignment"][0, 1] > .9
    assert distant["coassignment"][0, 1] < .01


def test_sensor_reproducibility_and_reserved_seeds():
    a, b = scenario(0, "high_clutter"), scenario(0, "high_clutter")
    np.testing.assert_array_equal(a["truth"], b["truth"])
    for za, zb in zip(a["measurements"], b["measurements"]):
        np.testing.assert_array_equal(za, zb)
    tracker = HistoryTracker(BirthAdapter({"learner": "fixed"}, a["sensor"]))
    assert tracker.detection_probability == .7
    assert tracker.birth_model.clutter_intensity == 12/7000
    assert tracker.survival_probability == a["sensor"].survival
    with pytest.raises(ValueError, match="reserved"):
        scenario(300, "stationary2")


def test_grid_sizes_and_admission_after_complete_scan():
    grid = configurations()
    assert [sum(c["learner"] == name for c in grid) for name in ["dp", "finite", "kde"]] == [9, 45, 9]
    adapter = BirthAdapter({"learner": "kde", "bandwidth": 1., "residual": .2}, Sensor())
    tracker = HistoryTracker(adapter)
    for t in range(4):
        tracker.step(np.array([[float(t), 0.]]))
        assert len(adapter.learner.kernels) == 0
    tracker.step(np.array([[4., 0.]]))
    assert len(adapter.learner.kernels) == 1
    evidence = tracker.admitted[0]
    assert evidence.first_detection == 0 and evidence.end_scan == 4
    assert evidence.observations == tuple((float(t), 0.) for t in range(5))
    for t in range(5, 10):
        tracker.step(np.array([[float(t), 0.]]))
    assert len(adapter.learner.kernels) == 1


def test_common_replay_independent_of_learner():
    a, stream_a = replay_stream(1, "stationary2")
    b, stream_b = replay_stream(1, "stationary2")
    assert a is b and stream_a is stream_b
    assert all(history.admissible for history, _ in stream_a)
    assert len({history.label for history, _ in stream_a}) == len(stream_a)
