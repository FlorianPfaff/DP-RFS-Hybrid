"""Characterize limitations without enshrining them as desired filter behavior."""

import math

import pytest

from dp_rfs_hybrid.birth_model_audit import (
    confirmation_time_case,
    diffuse_evidence_cases,
    no_spatial_information_case,
    posterior_feedback_cases,
    valid_identity_controls,
)


def test_same_posterior_can_have_different_likelihoods() -> None:
    first, second = posterior_feedback_cases()
    for key in ("posterior_mean", "posterior_variance", "overlap_plus_minus_ratio"):
        assert first[key] == pytest.approx(second[key])
    assert first["posterior_mean"] == pytest.approx(1.0)
    assert first["posterior_variance"] == pytest.approx(0.5)
    assert first["overlap_plus_minus_ratio"] == pytest.approx(math.exp(8.0 / 3.0))
    assert first["likelihood_plus_minus_ratio"] == pytest.approx(math.exp(4.0))
    assert second["likelihood_plus_minus_ratio"] == pytest.approx(1.0)


def test_constant_spatial_likelihood_does_not_equal_posterior_overlap() -> None:
    result = no_spatial_information_case()
    assert result["likelihood_plus_minus_ratio"] == 1.0
    assert result["overlap_plus_minus_ratio"] == pytest.approx(math.exp(4.0))


def test_diffuse_observation_merge_has_wrong_low_information_limit() -> None:
    rows = diffuse_evidence_cases()
    for row in rows:
        assert row["final_atoms"] == 1
        assert row["selected_atom"] == 0
        assert row["merged_region_variance"] == pytest.approx(
            (10.0 + row["observation_variance"]) / 11.0
        )
    assert rows[-1]["merged_region_variance"] == pytest.approx(910.0)
    assert rows[-1]["likelihood_variance_two_over_one"] > 0.9999


def test_overlap_and_moment_merge_have_valid_restricted_interpretations() -> None:
    result = valid_identity_controls()
    assert result["independent_observation_overlap_ratio"] == pytest.approx(
        result["independent_observation_likelihood_ratio"]
    )
    assert result["diffuse_prior_overlap_ratio"] == pytest.approx(
        result["independent_observation_likelihood_ratio"], rel=1e-6
    )
    assert result["moment_merge_mean"] == pytest.approx(1.0)
    assert result["moment_merge_variance"] == pytest.approx(4.0)


def test_tracker_passes_confirmation_state_not_birth_state() -> None:
    result = confirmation_time_case()
    assert result["confirmed_labels"] == 1
    assert result["initial_position"] == pytest.approx(0.0)
    assert result["learned_region_position"] == pytest.approx(2.0)
    assert result["learned_region_position"] == result["confirmation_position"]
