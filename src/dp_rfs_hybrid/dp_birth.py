"""Truncated Dirichlet-process birth model for unexplained measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gaussian import GaussianState


@dataclass
class BirthAtom:
    """Reusable Gaussian birth atom."""

    state: GaussianState
    count: float = 1.0
    last_updated: int = 0


@dataclass(frozen=True)
class BirthDecision:
    """Result of testing one measurement against the DP birth model."""

    accepted: bool
    branch: str
    odds: float
    atom_index: int | None = None
    state: GaussianState | None = None


@dataclass
class DirichletProcessBirthModel:
    """Finite active approximation to a DP Gaussian birth model.

    Existing atoms receive predictive mass proportional to ``count``. A new atom
    receives mass proportional to ``alpha`` under the base Gaussian state. The
    best DP-birth explanation is accepted only when its odds against clutter
    exceed ``odds_threshold``.
    """

    alpha: float
    base_state: GaussianState
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    clutter_intensity: float
    birth_probability: float = 0.8
    odds_threshold: float = 10.0
    max_atoms: int = 16
    prune_below_count: float = 0.05
    atoms: list[BirthAtom] = field(default_factory=list)
    scan_index: int = 0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.clutter_intensity <= 0:
            raise ValueError("clutter_intensity must be positive")
        if not 0.0 < self.birth_probability <= 1.0:
            raise ValueError("birth_probability must be in (0, 1]")
        if self.odds_threshold <= 0:
            raise ValueError("odds_threshold must be positive")
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        self.measurement_matrix = np.asarray(self.measurement_matrix, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)

    def score_existing_atoms(self, measurement: np.ndarray) -> list[float]:
        return [
            atom.count
            * atom.state.likelihood(
                measurement,
                self.measurement_matrix,
                self.measurement_noise,
            )
            for atom in self.atoms
        ]

    def score_new_atom(self, measurement: np.ndarray) -> float:
        return self.alpha * self.base_state.likelihood(
            measurement,
            self.measurement_matrix,
            self.measurement_noise,
        )

    def decide(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> BirthDecision:
        """Score a measurement without modifying model state."""

        measurement_vec = np.asarray(measurement, dtype=float)
        existing_scores = self.score_existing_atoms(measurement_vec)
        new_score = self.score_new_atom(measurement_vec)

        branch = "new"
        atom_index = None
        best_score = new_score
        if existing_scores:
            best_existing_index = int(np.argmax(existing_scores))
            best_existing_score = existing_scores[best_existing_index]
            if best_existing_score >= best_score:
                branch = "existing"
                atom_index = best_existing_index
                best_score = best_existing_score

        odds = self.birth_probability * best_score / self.clutter_intensity
        return BirthDecision(
            accepted=bool(odds > self.odds_threshold),
            branch=branch if odds > self.odds_threshold else "clutter",
            odds=float(odds),
            atom_index=atom_index,
        )

    def process(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> BirthDecision:
        """Update the DP birth model if the measurement is accepted as a birth."""

        measurement_vec = np.asarray(measurement, dtype=float)
        decision = self.decide(measurement_vec)
        if not decision.accepted:
            return decision

        self.scan_index += 1
        if decision.branch == "existing":
            assert decision.atom_index is not None
            atom = self.atoms[decision.atom_index]
            posterior, _ = atom.state.update(
                measurement_vec,
                self.measurement_matrix,
                self.measurement_noise,
            )
            atom.state = posterior
            atom.count += 1.0
            atom.last_updated = self.scan_index
            state = posterior
            atom_index = decision.atom_index
        else:
            posterior, _ = self.base_state.update(
                measurement_vec,
                self.measurement_matrix,
                self.measurement_noise,
            )
            self.atoms.append(BirthAtom(posterior, count=1.0, last_updated=self.scan_index))
            atom_index = len(self.atoms) - 1
            state = posterior

        self.prune()
        return BirthDecision(
            accepted=True,
            branch=decision.branch,
            odds=decision.odds,
            atom_index=atom_index,
            state=state,
        )

    def decay_counts(self, retention: float = 0.995) -> None:
        """Apply mild forgetting so stale atoms can eventually be pruned."""

        if not 0.0 < retention <= 1.0:
            raise ValueError("retention must be in (0, 1]")
        for atom in self.atoms:
            atom.count *= retention
        self.prune()

    def prune(self) -> None:
        self.atoms = [atom for atom in self.atoms if atom.count >= self.prune_below_count]
        if len(self.atoms) > self.max_atoms:
            self.atoms.sort(key=lambda atom: atom.count, reverse=True)
            self.atoms = self.atoms[: self.max_atoms]
