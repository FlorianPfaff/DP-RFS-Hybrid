"""Truncated Dirichlet-process birth model for unexplained measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gaussian import GaussianState, gaussian_pdf


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
    clutter_intensity: float | None = None
    birth_density: float | None = None
    birth_intensity: float | None = None
    existence_probability: float | None = None


@dataclass
class DirichletProcessBirthModel:
    """Finite active approximation to a DP Gaussian birth-density model.

    The DP layer models a normalized posterior-predictive birth density. The
    scalar ``birth_rate`` supplies the separate RFS birth mass/intensity scale.
    This mirrors the clutter convention ``kappa(z) = lambda_C c(z)`` and avoids
    using occupied DP component counts as a target-cardinality proxy.

    Existing atoms receive predictive mass proportional to ``count``. A new atom
    receives mass proportional to ``alpha`` under the base Gaussian state. The
    accepted birth intensity is

    ``birth_rate * posterior_predictive_birth_density(z)``.

    If ``birth_rate`` is left as ``None``, the implementation uses
    ``alpha * birth_probability`` for backwards-compatible default behavior.
    """

    alpha: float
    base_state: GaussianState
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray
    clutter_intensity: float
    birth_probability: float = 0.8
    odds_threshold: float = 10.0
    birth_rate: float | None = None
    max_atoms: int = 16
    prune_below_count: float = 0.05
    recluster_confirmed_states: bool = True
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
        if self.birth_rate is None:
            self.birth_rate = self.alpha * self.birth_probability
        if self.birth_rate <= 0.0:
            raise ValueError("birth_rate must be positive")
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        self.measurement_matrix = np.asarray(self.measurement_matrix, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)

    @property
    def total_count(self) -> float:
        """Return the active DP birth-atom count mass."""

        return float(sum(atom.count for atom in self.atoms))

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

    def predictive_birth_density(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
    ) -> float:
        """Evaluate the normalized DP posterior-predictive birth density."""

        measurement_vec = np.asarray(measurement, dtype=float)
        existing_scores = self.score_existing_atoms(measurement_vec)
        new_score = self.score_new_atom(measurement_vec)
        normalizer = self.alpha + self.total_count
        return float((sum(existing_scores) + new_score) / normalizer)

    def birth_intensity(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
    ) -> float:
        """Evaluate the measurement-space birth intensity used in odds tests."""

        assert self.birth_rate is not None
        return float(self.birth_rate * self.predictive_birth_density(measurement))

    def decide(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        clutter_intensity: float | None = None,
    ) -> BirthDecision:
        """Score a measurement without modifying model state."""

        measurement_vec = np.asarray(measurement, dtype=float)
        resolved_clutter_intensity = self._resolve_clutter_intensity(clutter_intensity)
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

        normalizer = self.alpha + self.total_count
        birth_density = float((sum(existing_scores) + new_score) / normalizer)
        assert self.birth_rate is not None
        birth_intensity = float(self.birth_rate * birth_density)
        odds = birth_intensity / resolved_clutter_intensity
        accepted = bool(odds > self.odds_threshold)
        return BirthDecision(
            accepted=accepted,
            branch=branch if accepted else "clutter",
            odds=float(odds),
            atom_index=atom_index,
            clutter_intensity=resolved_clutter_intensity,
            birth_density=birth_density,
            birth_intensity=birth_intensity,
            existence_probability=self.birth_probability,
        )

    def birth_state_from_decision(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        decision: BirthDecision,
    ) -> GaussianState | None:
        """Create a tentative birth state without updating DP atoms.

        This supports delayed birth learning: a measurement can initialize a
        tentative Bernoulli track immediately, while the DP birth density is only
        updated later if the track is confirmed by the RFS layer.
        """

        if not decision.accepted:
            return None
        measurement_vec = np.asarray(measurement, dtype=float)
        if decision.branch == "existing":
            if decision.atom_index is None:
                raise ValueError("existing-atom birth decision must include atom_index")
            posterior, _ = self.atoms[decision.atom_index].state.update(
                measurement_vec,
                self.measurement_matrix,
                self.measurement_noise,
            )
            return posterior
        posterior, _ = self.base_state.update(
            measurement_vec,
            self.measurement_matrix,
            self.measurement_noise,
        )
        return posterior

    def learn_confirmed_state(self, state: GaussianState, count: float = 1.0) -> int:
        """Update the DP birth model from confirmed birth evidence.

        Confirmed evidence is reclustered in state space rather than appended
        blindly. Existing atoms are scored by the overlap between the confirmed
        Gaussian state and the atom Gaussian. The residual/new-atom branch is
        scored by the overlap with the base state. If an existing atom wins, the
        confirmed state is merged into that atom with count-weighted moments.
        """

        if count <= 0.0:
            raise ValueError("count must be positive")
        self._validate_confirmed_state(state)
        self.scan_index += 1

        existing_scores = self.score_existing_confirmed_states(state)
        new_score = self.score_new_confirmed_state(state)
        if (
            self.recluster_confirmed_states
            and existing_scores
            and max(existing_scores) >= new_score
        ):
            atom_index = int(np.argmax(existing_scores))
            atom = self.atoms[atom_index]
            atom.state = self._merge_gaussian_states(
                old_state=atom.state,
                old_count=atom.count,
                confirmed_state=state,
                confirmed_count=count,
            )
            atom.count += count
            atom.last_updated = self.scan_index
            updated_atom = atom
        else:
            updated_atom = BirthAtom(state=state, count=count, last_updated=self.scan_index)
            self.atoms.append(updated_atom)

        self.prune()
        for atom_index, atom in enumerate(self.atoms):
            if atom is updated_atom:
                return atom_index
        return -1

    def score_existing_confirmed_states(self, state: GaussianState) -> list[float]:
        """Score confirmed birth evidence against active birth atoms."""

        self._validate_confirmed_state(state)
        return [
            atom.count
            * gaussian_pdf(
                state.mean,
                atom.state.mean,
                atom.state.covariance + state.covariance,
            )
            for atom in self.atoms
        ]

    def score_new_confirmed_state(self, state: GaussianState) -> float:
        """Score confirmed birth evidence against the residual DP branch."""

        self._validate_confirmed_state(state)
        return self.alpha * gaussian_pdf(
            state.mean,
            self.base_state.mean,
            self.base_state.covariance + state.covariance,
        )

    def process(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        clutter_intensity: float | None = None,
    ) -> BirthDecision:
        """Update the DP birth model if the measurement is accepted as a birth."""

        measurement_vec = np.asarray(measurement, dtype=float)
        decision = self.decide(measurement_vec, clutter_intensity=clutter_intensity)
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
            clutter_intensity=decision.clutter_intensity,
            birth_density=decision.birth_density,
            birth_intensity=decision.birth_intensity,
            existence_probability=decision.existence_probability,
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

    def _resolve_clutter_intensity(self, clutter_intensity: float | None) -> float:
        if clutter_intensity is None:
            return float(self.clutter_intensity)
        clutter_intensity = float(clutter_intensity)
        if clutter_intensity <= 0.0:
            raise ValueError("clutter_intensity must be positive")
        return clutter_intensity

    def _validate_confirmed_state(self, state: GaussianState) -> None:
        if state.dim != self.base_state.dim:
            raise ValueError("confirmed state dimension must match base state dimension")

    @staticmethod
    def _merge_gaussian_states(
        old_state: GaussianState,
        old_count: float,
        confirmed_state: GaussianState,
        confirmed_count: float,
    ) -> GaussianState:
        new_count = old_count + confirmed_count
        new_mean = (
            old_count * old_state.mean + confirmed_count * confirmed_state.mean
        ) / new_count
        old_delta = old_state.mean - new_mean
        confirmed_delta = confirmed_state.mean - new_mean
        new_covariance = (
            old_count
            * (old_state.covariance + np.outer(old_delta, old_delta))
            + confirmed_count
            * (confirmed_state.covariance + np.outer(confirmed_delta, confirmed_delta))
        ) / new_count
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        return GaussianState(mean=new_mean, covariance=new_covariance)
