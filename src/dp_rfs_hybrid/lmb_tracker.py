"""A compact labeled multi-Bernoulli-style tracker using DP births."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dp_birth import DirichletProcessBirthModel
from .dp_clutter import DirichletProcessClutterModel, FixedGaussianMixtureClutterModel
from .gaussian import GaussianState

ClutterModel = DirichletProcessClutterModel | FixedGaussianMixtureClutterModel


@dataclass
class Track:
    label: int
    state: GaussianState
    existence: float
    age: int = 0
    missed: int = 0


@dataclass(frozen=True)
class StepSummary:
    assignments: list[tuple[int, int]]
    births: list[int]
    clutter: list[int]
    missed_tracks: list[int]
    clutter_updates: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class LabeledMultiBernoulliTracker:
    """Small RFS-style tracker for experimenting with DP nuisance models.

    The tracker keeps labels and Bernoulli existence probabilities in the RFS
    layer. The DP birth model handles reusable birth regions. If an optional
    clutter model is supplied, its intensity replaces the scalar clutter
    intensity in association odds, existence updates, and birth decisions.
    RFS-style clutter responsibilities are then fed back into adaptive clutter
    models as fractional observations.
    """

    transition_matrix: np.ndarray
    process_noise: np.ndarray
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    birth_model: DirichletProcessBirthModel
    clutter_model: ClutterModel | None = None
    survival_probability: float = 0.98
    detection_probability: float = 0.9
    association_threshold: float = 5.0
    prune_below_existence: float = 0.05
    max_tracks: int = 64
    clutter_responsibility_learning_rate: float = 1.0
    min_clutter_responsibility_to_learn: float = 0.0
    tracks: list[Track] = field(default_factory=list)
    next_label: int = 1

    def __post_init__(self) -> None:
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=float)
        self.process_noise = np.asarray(self.process_noise, dtype=float)
        self.measurement_matrix = np.asarray(self.measurement_matrix, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)
        if not 0.0 < self.survival_probability <= 1.0:
            raise ValueError("survival_probability must be in (0, 1]")
        if not 0.0 < self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in (0, 1]")
        if not 0.0 <= self.clutter_responsibility_learning_rate <= 1.0:
            raise ValueError("clutter_responsibility_learning_rate must be in [0, 1]")
        if not 0.0 <= self.min_clutter_responsibility_to_learn <= 1.0:
            raise ValueError("min_clutter_responsibility_to_learn must be in [0, 1]")

    def predict(self) -> None:
        for track in self.tracks:
            track.state = track.state.predict(self.transition_matrix, self.process_noise)
            track.existence *= self.survival_probability
            track.age += 1

    def step(self, measurements: np.ndarray | list[list[float]]) -> StepSummary:
        measurement_array = np.asarray(measurements, dtype=float)
        if measurement_array.size == 0:
            measurement_array = np.empty((0, self.measurement_matrix.shape[0]))
        if measurement_array.ndim != 2:
            raise ValueError("measurements must be a two-dimensional array")

        self.predict()
        assignments = self._greedy_assign(measurement_array)
        assigned_track_indices = {track_index for track_index, _ in assignments}
        assigned_measurement_indices = {measurement_index for _, measurement_index in assignments}
        clutter_evidence: list[tuple[int, np.ndarray, float]] = []

        for track_index, measurement_index in assignments:
            track = self.tracks[track_index]
            measurement = measurement_array[measurement_index]
            posterior, likelihood = track.state.update(
                measurement,
                self.measurement_matrix,
                self.measurement_noise,
            )
            clutter_intensity = self._clutter_intensity(measurement)
            numerator = track.existence * self.detection_probability * likelihood
            denominator = clutter_intensity + numerator
            track.existence = float(min(0.999, numerator / max(denominator, 1e-300)))
            track.state = posterior
            track.missed = 0
            clutter_responsibility = self._clutter_responsibility_from_competing_weights(
                clutter_intensity,
                numerator,
            )
            clutter_evidence.append((measurement_index, measurement, clutter_responsibility))

        missed_tracks: list[int] = []
        for track_index, track in enumerate(self.tracks):
            if track_index in assigned_track_indices:
                continue
            missed_tracks.append(track.label)
            denominator = 1.0 - track.existence * self.detection_probability
            if denominator <= 1e-12:
                track.existence = self.prune_below_existence
            else:
                track.existence = track.existence * (1.0 - self.detection_probability) / denominator
            track.missed += 1

        births: list[int] = []
        clutter: list[int] = []
        for measurement_index, measurement in enumerate(measurement_array):
            if measurement_index in assigned_measurement_indices:
                continue
            clutter_intensity = self._clutter_intensity(measurement)
            decision = self.birth_model.process(
                measurement,
                clutter_intensity=clutter_intensity,
            )
            clutter_responsibility = self._clutter_responsibility_from_birth_odds(decision.odds)
            clutter_evidence.append((measurement_index, measurement, clutter_responsibility))
            if decision.accepted and decision.state is not None:
                label = self.next_label
                self.next_label += 1
                births.append(label)
                self.tracks.append(
                    Track(
                        label=label,
                        state=decision.state,
                        existence=self.birth_model.birth_probability,
                    )
                )
            else:
                clutter.append(measurement_index)

        clutter_updates = self._update_clutter_model(clutter_evidence)
        assignment_labels = [
            (self.tracks[track_index].label, measurement_index)
            for track_index, measurement_index in assignments
        ]
        self.prune()
        self.birth_model.decay_counts()
        if self.clutter_model is not None:
            self.clutter_model.decay_counts()
        return StepSummary(
            assignments=assignment_labels,
            births=births,
            clutter=clutter,
            missed_tracks=missed_tracks,
            clutter_updates=clutter_updates,
        )

    def estimates(self, existence_threshold: float = 0.5) -> list[Track]:
        return [track for track in self.tracks if track.existence >= existence_threshold]

    def _greedy_assign(self, measurements: np.ndarray) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for measurement_index, measurement in enumerate(measurements):
                likelihood = track.state.likelihood(
                    measurement,
                    self.measurement_matrix,
                    self.measurement_noise,
                )
                odds = (
                    track.existence
                    * self.detection_probability
                    * likelihood
                    / self._clutter_intensity(measurement)
                )
                if odds > self.association_threshold:
                    candidates.append((float(odds), track_index, measurement_index))

        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _, track_index, measurement_index in candidates:
            if track_index in used_tracks or measurement_index in used_measurements:
                continue
            assignments.append((track_index, measurement_index))
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)
        return assignments

    def prune(self) -> None:
        self.tracks = [
            track for track in self.tracks if track.existence >= self.prune_below_existence
        ]
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda track: track.existence, reverse=True)
            self.tracks = self.tracks[: self.max_tracks]

    def _clutter_intensity(self, measurement: np.ndarray) -> float:
        if self.clutter_model is None:
            return float(self.birth_model.clutter_intensity)
        return max(float(self.clutter_model.intensity(measurement)), 1e-300)

    @staticmethod
    def _clutter_responsibility_from_competing_weights(
        clutter_weight: float,
        target_weight: float,
    ) -> float:
        denominator = clutter_weight + target_weight
        if denominator <= 0.0:
            return 1.0
        return float(np.clip(clutter_weight / denominator, 0.0, 1.0))

    @staticmethod
    def _clutter_responsibility_from_birth_odds(birth_odds: float) -> float:
        birth_odds = max(float(birth_odds), 0.0)
        return float(1.0 / (1.0 + birth_odds))

    def _learned_clutter_responsibility(self, responsibility: float) -> float:
        if responsibility < self.min_clutter_responsibility_to_learn:
            return 0.0
        return float(self.clutter_responsibility_learning_rate * responsibility)

    def _update_clutter_model(
        self,
        clutter_evidence: list[tuple[int, np.ndarray, float]],
    ) -> list[tuple[int, float]]:
        if self.clutter_model is None:
            return []
        updates: list[tuple[int, float]] = []
        for measurement_index, measurement, responsibility in clutter_evidence:
            learned_responsibility = self._learned_clutter_responsibility(responsibility)
            self.clutter_model.update(measurement, responsibility=learned_responsibility)
            updates.append((measurement_index, learned_responsibility))
        return updates
