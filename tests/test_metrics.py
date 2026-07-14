import numpy as np
import pytest

from dp_rfs_hybrid import gospa


def test_gospa_is_zero_for_identical_sets() -> None:
    points = np.array([[0.0, 0.0], [2.0, 1.0]])

    result = gospa(points, points)

    assert result.distance == pytest.approx(0.0)
    assert result.matched_count == 2
    assert result.missed_count == 0
    assert result.false_count == 0


def test_gospa_decomposes_missed_and_false_targets() -> None:
    truth = np.array([[0.0, 0.0], [20.0, 0.0]])
    estimates = np.array([[1.0, 0.0], [-20.0, 0.0]])

    result = gospa(truth, estimates, cutoff=10.0, order=2.0, alpha=2.0)

    assert result.localization == pytest.approx(1.0)
    assert result.missed == pytest.approx(50.0)
    assert result.false == pytest.approx(50.0)
    assert result.powered_distance == pytest.approx(101.0)
    assert result.matched_count == 1
    assert result.missed_count == 1
    assert result.false_count == 1


def test_gospa_handles_empty_sets() -> None:
    result = gospa(np.empty((0, 2)), np.array([[1.0, 2.0]]), cutoff=10.0)

    assert result.false == pytest.approx(50.0)
    assert result.false_count == 1
    assert result.missed_count == 0
