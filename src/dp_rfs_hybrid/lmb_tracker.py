"""A compact labeled multi-Bernoulli-style tracker using DP births."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dp_birth import DirichletProcessBirthModel
from .gaussian import GaussianState


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


@dataclass
class LabeledMultiBernoulliTracker:
    """Small RFS-style tracker for experimenting with DP birth decisions."""

    transition_matrix: np.ndarray
    process_noise: np.ndarray
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    birth_model: DirichletProcessBirthModel
    survival_probability: float = 0.98
    detection_probability: float = 0.9
    association_threshold: float = 5.0
    prune_below_existence: float = 0.05
    max_tracks: int = 64
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

        for track_index, measurement_index in assignments:
            track = self.tracks[track_index]
            posterior, likelihood = track.state.update(
                measurement_array[measurement_index],
                self.measurement_matrix,
                self.measurement_noise,
            )
            numerator = track.existence * self.detection_probability * likelihood
            denominator = self.birth_model.clutter_intensity + numerator
            track.existence = float(min(0.999, numerator / denominator))
            track.state = posterior
            track.missed = 0

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
            decision = self.birth_model.process(measurement)
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

        assignment_labels = [
            (self.tracks[track_index].label, measurement_index)
            for track_index, measurement_index in assignments
        ]
        self.prune()
        self.birth_model.decay_counts()
        return StepSummary(
            assignments=assignment_labels,
            births=births,
            clutter=clutter,
            missed_tracks=missed_tracks,
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
                    / self.birth_model.clutter_intensity
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
