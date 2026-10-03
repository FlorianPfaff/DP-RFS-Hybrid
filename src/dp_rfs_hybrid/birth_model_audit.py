"""Deterministic diagnostics of the existing birth learner, not a replacement.

These examples distinguish Gaussian overlap identities from using a track
posterior as an independent likelihood. They do not measure tracking accuracy.
"""

from __future__ import annotations

import numpy as np

from .dp_birth import BirthAtom, DirichletProcessBirthModel
from .gaussian import GaussianState, gaussian_pdf
from .lmb_tracker import LabeledMultiBernoulliTracker


def _scalar_state(mean: float, variance: float) -> GaussianState:
    return GaussianState(np.array([mean]), np.array([[variance]]))


def _two_region_model() -> DirichletProcessBirthModel:
    model = DirichletProcessBirthModel(
        alpha=0.1,
        base_state=_scalar_state(0.0, 100.0),
        measurement_matrix=np.eye(1),
        measurement_noise=np.eye(1),
        clutter_intensity=0.01,
        birth_rate=1.0,
    )
    model.atoms = [
        BirthAtom(_scalar_state(-2.0, 1.0)),
        BirthAtom(_scalar_state(2.0, 1.0)),
    ]
    return model


def posterior_feedback_cases() -> list[dict[str, float | str]]:
    """Identical posteriors can conceal different Gaussian likelihoods."""

    model = _two_region_model()
    cases: list[dict[str, float | str]] = []
    for name, prior_mean, observation in (
        ("prior_zero_observation_two", 0.0, 2.0),
        ("prior_two_observation_zero", 2.0, 0.0),
    ):
        prior = _scalar_state(prior_mean, 1.0)
        posterior, _ = prior.update(np.array([observation]), np.eye(1), np.eye(1))
        scores = model.score_existing_confirmed_states(posterior)
        evidence = [atom.state.likelihood([observation], np.eye(1), np.eye(1))
                    for atom in model.atoms]
        cases.append({
            "case": name,
            "prior_mean": prior_mean,
            "prior_variance": 1.0,
            "observation": observation,
            "observation_variance": 1.0,
            "posterior_mean": float(posterior.mean[0]),
            "posterior_variance": float(posterior.covariance[0, 0]),
            "overlap_plus_minus_ratio": scores[1] / scores[0],
            "likelihood_plus_minus_ratio": evidence[1] / evidence[0],
        })
    return cases


def no_spatial_information_case() -> dict[str, float]:
    """A constant spatial likelihood must not change assignment odds.

    This is an evidence-interface diagnostic, not a claim that the tracker
    confirms an empty track. Existence evidence can be conditioned on separately.
    """

    model = _two_region_model()
    posterior_equal_to_prior = _scalar_state(2.0, 1.0)
    scores = model.score_existing_confirmed_states(posterior_equal_to_prior)
    return {
        "prior_plus_minus_ratio": 1.0,
        "likelihood_plus_minus_ratio": 1.0,
        "overlap_plus_minus_ratio": scores[1] / scores[0],
    }


def diffuse_evidence_cases() -> list[dict[str, float | int]]:
    """Test the alternative interpretation of the input as a noisy observation.

    Under y | x ~ N(x, R), the likelihood ratio for intrinsic region variances
    2 versus 1 tends to one as R grows. The current moment merge instead adds
    observation uncertainty to the reusable region's covariance.
    """

    rows: list[dict[str, float | int]] = []
    for variance in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        model = _two_region_model()
        model.atoms = [BirthAtom(_scalar_state(0.0, 1.0), count=10.0)]
        index = model.learn_confirmed_state(_scalar_state(0.0, variance))
        likelihood_one = gaussian_pdf([0.0], [0.0], [[1.0 + variance]])
        likelihood_two = gaussian_pdf([0.0], [0.0], [[2.0 + variance]])
        rows.append({
            "observation_variance": variance,
            "initial_region_variance": 1.0,
            "initial_count": 10.0,
            "selected_atom": index,
            "final_atoms": len(model.atoms),
            "merged_region_variance": float(model.atoms[index].state.covariance[0, 0]),
            "likelihood_variance_two_over_one": likelihood_two / likelihood_one,
        })
    return rows


def valid_identity_controls() -> dict[str, float]:
    """Positive controls: overlap and moment matching are valid identities."""

    model = _two_region_model()
    independent_observation = _scalar_state(2.0, 1.0)
    scores = model.score_existing_confirmed_states(independent_observation)
    likelihoods = [atom.state.likelihood([2.0], np.eye(1), np.eye(1))
                   for atom in model.atoms]
    diffuse_prior = _scalar_state(0.0, 1e8)
    posterior, _ = diffuse_prior.update([2.0], np.eye(1), np.eye(1))
    diffuse_scores = model.score_existing_confirmed_states(posterior)
    merged = model._merge_gaussian_states(
        _scalar_state(0.0, 1.0), 2.0, _scalar_state(3.0, 4.0), 1.0,
    )
    return {
        "independent_observation_overlap_ratio": scores[1] / scores[0],
        "independent_observation_likelihood_ratio": likelihoods[1] / likelihoods[0],
        "diffuse_prior_overlap_ratio": diffuse_scores[1] / diffuse_scores[0],
        "moment_merge_mean": float(merged.mean[0]),
        "moment_merge_variance": float(merged.covariance[0, 0]),
    }


def confirmation_time_case() -> dict[str, float | int]:
    """Follow the existing tracker for two deterministic constant-velocity scans."""

    model = DirichletProcessBirthModel(
        alpha=1.0,
        base_state=GaussianState(np.array([0.0, 2.0]), np.diag([1.0, 0.01])),
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_noise=np.array([[0.01]]),
        clutter_intensity=0.001,
        birth_probability=0.35,
        birth_rate=1.0,
        odds_threshold=0.01,
    )
    tracker = LabeledMultiBernoulliTracker(
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        process_noise=1e-6 * np.eye(2),
        measurement_matrix=model.measurement_matrix,
        measurement_noise=model.measurement_noise,
        birth_model=model,
        survival_probability=0.98,
        detection_probability=0.9,
        association_threshold=0.2,
        delayed_birth_learning=True,
        birth_confirmation_age=1,
        birth_confirmation_existence=0.6,
    )
    tracker.step([[0.0]])
    initial_position = float(tracker.tracks[0].state.mean[0])
    summary = tracker.step([[2.0]])
    return {
        "initial_position": initial_position,
        "confirmation_position": float(tracker.tracks[0].state.mean[0]),
        "learned_region_position": float(model.atoms[0].state.mean[0]),
        "confirmed_labels": len(summary.confirmed_births),
        "elapsed_scans": 1,
    }


def run_birth_model_audit() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study": "dp-specific-feasibility-stage-one",
        "interpretation": "Deterministic model diagnostics; not tracking benchmarks.",
        "posterior_feedback": posterior_feedback_cases(),
        "no_spatial_information": no_spatial_information_case(),
        "diffuse_evidence": diffuse_evidence_cases(),
        "positive_controls": valid_identity_controls(),
        "confirmation_time": confirmation_time_case(),
    }
