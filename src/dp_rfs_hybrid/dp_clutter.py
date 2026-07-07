"""Dirichlet-process clutter density model for RFS-style trackers.

The model deliberately estimates a *normalized clutter density* in measurement
space. The Poisson clutter rate is represented separately by ``rate`` so that the
RFS intensity can be evaluated as ``kappa(z) = rate * c(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gaussian import gaussian_pdf


@dataclass
class ClutterAtom:
    """Reusable Gaussian component for structured clutter in measurement space."""

    mean: np.ndarray
    covariance: np.ndarray
    count: float = 1.0
    last_updated: int = 0

    def __post_init__(self) -> None:
        self.mean = _as_vector(self.mean)
        self.covariance = _as_square_matrix(self.covariance, self.mean.size)
        if self.count <= 0.0:
            raise ValueError("count must be positive")

    def likelihood(self, measurement: np.ndarray) -> float:
        return gaussian_pdf(measurement, self.mean, self.covariance)

    def update_mean(self, measurement: np.ndarray, weight: float) -> None:
        """Apply a weighted online mean update while keeping covariance fixed."""

        if weight <= 0.0:
            return
        new_count = self.count + weight
        self.mean = (self.count * self.mean + weight * measurement) / new_count
        self.count = new_count


@dataclass(frozen=True)
class ClutterUpdate:
    """Diagnostic result of a weighted DP clutter update."""

    responsibility: float
    branch: str
    atom_index: int | None
    density_before_update: float
    intensity_before_update: float


@dataclass
class DirichletProcessClutterModel:
    """Finite active approximation to a DP mixture clutter density.

    The model is intended to be updated from RFS posterior responsibilities rather
    than from hard pre-clustered clutter observations. A measurement with clutter
    responsibility ``r`` contributes fractional count ``r`` to either the most
    likely existing atom or to a newly instantiated residual atom.
    """

    alpha: float
    base_mean: np.ndarray
    base_covariance: np.ndarray
    rate: float
    max_atoms: int = 32
    prune_below_count: float = 0.05
    atoms: list[ClutterAtom] = field(default_factory=list)
    update_index: int = 0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.rate <= 0.0:
            raise ValueError("rate must be positive")
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        self.base_mean = _as_vector(self.base_mean)
        self.base_covariance = _as_square_matrix(self.base_covariance, self.base_mean.size)

    @property
    def total_count(self) -> float:
        return float(sum(atom.count for atom in self.atoms))

    def base_likelihood(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> float:
        measurement_vec = _as_vector(measurement)
        return gaussian_pdf(measurement_vec, self.base_mean, self.base_covariance)

    def component_scores(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> tuple[list[float], float]:
        """Return existing-atom scores and the residual new-atom score."""

        measurement_vec = _as_vector(measurement)
        existing_scores = [atom.count * atom.likelihood(measurement_vec) for atom in self.atoms]
        new_score = self.alpha * self.base_likelihood(measurement_vec)
        return existing_scores, new_score

    def density(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> float:
        """Evaluate the posterior-predictive normalized clutter density."""

        existing_scores, new_score = self.component_scores(measurement)
        normalizer = self.alpha + self.total_count
        return float((sum(existing_scores) + new_score) / normalizer)

    def intensity(self, measurement: np.ndarray | list[float] | tuple[float, ...]) -> float:
        """Evaluate the Poisson clutter intensity ``kappa(z)``."""

        return float(self.rate * self.density(measurement))

    def update(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        responsibility: float = 1.0,
    ) -> ClutterUpdate:
        """Update the DP clutter model with a fractional clutter observation.

        Parameters
        ----------
        measurement:
            Measurement-space vector.
        responsibility:
            Posterior probability or soft weight that the measurement is clutter.
            Values near zero leave the model unchanged but still return predictive
            diagnostics.
        """

        measurement_vec = _as_vector(measurement)
        responsibility = float(responsibility)
        if not 0.0 <= responsibility <= 1.0:
            raise ValueError("responsibility must be in [0, 1]")

        density_before = self.density(measurement_vec)
        intensity_before = self.rate * density_before
        existing_scores, new_score = self.component_scores(measurement_vec)
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

        if responsibility <= 0.0:
            return ClutterUpdate(
                responsibility=responsibility,
                branch="ignored",
                atom_index=None,
                density_before_update=float(density_before),
                intensity_before_update=float(intensity_before),
            )

        self.update_index += 1
        if branch == "existing":
            assert atom_index is not None
            atom = self.atoms[atom_index]
            atom.update_mean(measurement_vec, responsibility)
            atom.last_updated = self.update_index
        else:
            self.atoms.append(
                ClutterAtom(
                    mean=measurement_vec.copy(),
                    covariance=self.base_covariance.copy(),
                    count=responsibility,
                    last_updated=self.update_index,
                )
            )
            atom_index = len(self.atoms) - 1

        self.prune()
        return ClutterUpdate(
            responsibility=responsibility,
            branch=branch,
            atom_index=atom_index,
            density_before_update=float(density_before),
            intensity_before_update=float(intensity_before),
        )

    def decay_counts(self, retention: float = 0.995) -> None:
        """Apply exponential forgetting to allow stale clutter atoms to disappear."""

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


def _as_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    return vector


def _as_square_matrix(value: np.ndarray | list[list[float]], dim: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dim, dim):
        raise ValueError(f"Expected a square matrix with shape ({dim}, {dim})")
    return matrix
