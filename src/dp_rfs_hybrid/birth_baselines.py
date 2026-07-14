"""Fixed and measurement-driven birth baselines for controlled comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dp_birth import BirthDecision
from .gaussian import GaussianState


@dataclass
class FixedGaussianMixtureBirthModel:
    """Non-learning Gaussian-mixture birth model.

    This model is useful both as a broad fixed-prior baseline and as an oracle
    whose components are placed at the true recurring birth regions.
    """

    weights: np.ndarray
    states: list[GaussianState]
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    clutter_intensity: float
    birth_rate: float
    birth_probability: float = 0.35
    odds_threshold: float = 0.02
    atoms: list[object] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.measurement_matrix = np.asarray(self.measurement_matrix, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)
        if self.weights.ndim != 1 or len(self.weights) != len(self.states):
            raise ValueError("weights must contain one entry per Gaussian state")
        if np.any(self.weights < 0.0) or not np.any(self.weights > 0.0):
            raise ValueError("weights must be nonnegative and have positive mass")
        self.weights = self.weights / np.sum(self.weights)
        if self.clutter_intensity <= 0.0:
            raise ValueError("clutter_intensity must be positive")
        if self.birth_rate <= 0.0:
            raise ValueError("birth_rate must be positive")
        if not 0.0 < self.birth_probability <= 1.0:
            raise ValueError("birth_probability must be in (0, 1]")
        if self.odds_threshold <= 0.0:
            raise ValueError("odds_threshold must be positive")

    def process(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        clutter_intensity: float | None = None,
    ) -> BirthDecision:
        measurement_vec = np.asarray(measurement, dtype=float)
        component_densities = np.asarray(
            [
                state.likelihood(
                    measurement_vec,
                    self.measurement_matrix,
                    self.measurement_noise,
                )
                for state in self.states
            ]
        )
        weighted_densities = self.weights * component_densities
        birth_density = float(np.sum(weighted_densities))
        birth_intensity = self.birth_rate * birth_density
        resolved_clutter = (
            self.clutter_intensity
            if clutter_intensity is None
            else float(clutter_intensity)
        )
        odds = birth_intensity / resolved_clutter
        accepted = bool(odds > self.odds_threshold)
        if not accepted:
            return BirthDecision(
                accepted=False,
                branch="clutter",
                odds=odds,
                clutter_intensity=resolved_clutter,
                birth_density=birth_density,
                birth_intensity=birth_intensity,
                existence_probability=self.birth_probability,
            )

        component_index = int(np.argmax(weighted_densities))
        state, _ = self.states[component_index].update(
            measurement_vec,
            self.measurement_matrix,
            self.measurement_noise,
        )
        return BirthDecision(
            accepted=True,
            branch="fixed",
            odds=odds,
            atom_index=component_index,
            state=state,
            clutter_intensity=resolved_clutter,
            birth_density=birth_density,
            birth_intensity=birth_intensity,
            existence_probability=self.birth_probability,
        )

    def decay_counts(self, retention: float = 0.995) -> None:
        """Keep the fixed model unchanged."""


@dataclass
class MeasurementDrivenBirthModel:
    """Greedy-LMB analogue of the published measurement-driven birth model.

    In a full GLMB implementation, the unassigned probability is marginalized
    over global association hypotheses. In this prototype, the greedy RFS
    backend supplies the unassigned measurement set. Each such measurement
    creates a tentative Bernoulli whose existence mass is normalized so the
    batch sum is at most ``expected_births`` and each component is capped by
    ``max_birth_probability``.
    """

    base_state: GaussianState
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    clutter_intensity: float
    expected_births: float = 0.15
    max_birth_probability: float = 0.15
    birth_probability: float = field(default=0.15, init=False)
    atoms: list[object] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.measurement_matrix = np.asarray(self.measurement_matrix, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)
        if self.clutter_intensity <= 0.0:
            raise ValueError("clutter_intensity must be positive")
        if self.expected_births <= 0.0:
            raise ValueError("expected_births must be positive")
        if not 0.0 < self.max_birth_probability <= 1.0:
            raise ValueError("max_birth_probability must be in (0, 1]")

    def begin_birth_batch(self, candidate_count: int) -> None:
        if candidate_count < 0:
            raise ValueError("candidate_count must be nonnegative")
        if candidate_count == 0:
            self.birth_probability = self.max_birth_probability
            return
        self.birth_probability = min(
            self.max_birth_probability,
            self.expected_births / candidate_count,
        )

    def process(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        clutter_intensity: float | None = None,
    ) -> BirthDecision:
        measurement_vec = np.asarray(measurement, dtype=float)
        state, likelihood = self.base_state.update(
            measurement_vec,
            self.measurement_matrix,
            self.measurement_noise,
        )
        resolved_clutter = (
            self.clutter_intensity
            if clutter_intensity is None
            else float(clutter_intensity)
        )
        odds = self.birth_probability / max(1.0 - self.birth_probability, 1e-12)
        return BirthDecision(
            accepted=True,
            branch="measurement_driven",
            odds=odds,
            state=state,
            clutter_intensity=resolved_clutter,
            birth_density=likelihood,
            birth_intensity=None,
            existence_probability=self.birth_probability,
        )

    def decay_counts(self, retention: float = 0.995) -> None:
        """Keep the measurement-driven model unchanged."""
