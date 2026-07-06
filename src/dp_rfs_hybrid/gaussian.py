"""Small linear-Gaussian utilities used by the prototype tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    return vector


def _as_matrix(value: np.ndarray | list[list[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix")
    return matrix


def gaussian_pdf(
    x: np.ndarray | list[float] | tuple[float, ...],
    mean: np.ndarray | list[float] | tuple[float, ...],
    covariance: np.ndarray | list[list[float]],
) -> float:
    """Evaluate a multivariate Gaussian density."""

    x_vec = _as_vector(x)
    mean_vec = _as_vector(mean)
    cov = _as_matrix(covariance)
    if x_vec.shape != mean_vec.shape:
        raise ValueError("x and mean must have the same shape")
    if cov.shape != (x_vec.size, x_vec.size):
        raise ValueError("covariance shape does not match vector dimension")

    jitter = 1e-9 * np.eye(x_vec.size)
    chol = np.linalg.cholesky(cov + jitter)
    delta = x_vec - mean_vec
    solved = np.linalg.solve(chol, delta)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    log_pdf = -0.5 * (x_vec.size * np.log(2.0 * np.pi) + log_det + solved @ solved)
    return float(np.exp(log_pdf))


@dataclass(frozen=True)
class GaussianState:
    """Mean and covariance for a Gaussian single-target state."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = _as_vector(self.mean)
        covariance = _as_matrix(self.covariance)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance shape must match mean dimension")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    def predict(self, transition_matrix: np.ndarray, process_noise: np.ndarray) -> "GaussianState":
        transition = _as_matrix(transition_matrix)
        noise = _as_matrix(process_noise)
        if transition.shape != (self.dim, self.dim):
            raise ValueError("transition matrix shape does not match state dimension")
        if noise.shape != (self.dim, self.dim):
            raise ValueError("process noise shape does not match state dimension")
        return GaussianState(
            mean=transition @ self.mean,
            covariance=transition @ self.covariance @ transition.T + noise,
        )

    def measurement_prediction(
        self,
        measurement_matrix: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        measurement = _as_matrix(measurement_matrix)
        noise = _as_matrix(measurement_noise)
        if measurement.shape[1] != self.dim:
            raise ValueError("measurement matrix shape does not match state dimension")
        if noise.shape != (measurement.shape[0], measurement.shape[0]):
            raise ValueError("measurement noise shape does not match measurement dimension")
        mean = measurement @ self.mean
        covariance = measurement @ self.covariance @ measurement.T + noise
        return mean, covariance

    def likelihood(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        measurement_matrix: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> float:
        predicted_mean, predicted_covariance = self.measurement_prediction(
            measurement_matrix,
            measurement_noise,
        )
        return gaussian_pdf(measurement, predicted_mean, predicted_covariance)

    def update(
        self,
        measurement: np.ndarray | list[float] | tuple[float, ...],
        measurement_matrix: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> tuple["GaussianState", float]:
        measurement_vec = _as_vector(measurement)
        measurement_model = _as_matrix(measurement_matrix)
        noise = _as_matrix(measurement_noise)
        predicted_measurement, innovation_covariance = self.measurement_prediction(
            measurement_model,
            noise,
        )
        if measurement_vec.shape != predicted_measurement.shape:
            raise ValueError("measurement shape does not match measurement model")

        gain = self.covariance @ measurement_model.T @ np.linalg.inv(innovation_covariance)
        innovation = measurement_vec - predicted_measurement
        posterior_mean = self.mean + gain @ innovation
        identity = np.eye(self.dim)
        posterior_covariance = (identity - gain @ measurement_model) @ self.covariance
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
        return GaussianState(posterior_mean, posterior_covariance), gaussian_pdf(
            measurement_vec,
            predicted_measurement,
            innovation_covariance,
        )
