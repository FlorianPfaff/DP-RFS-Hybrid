"""Prior-free Gaussian likelihood messages for selected association histories.

The conditioning on association and admission is intentional. These messages
are not a joint RFS posterior and do not correct selection or fragmentation.
"""

from dataclasses import dataclass

import numpy as np


def readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def logsumexp(values, axis=None):
    values = np.asarray(values)
    maximum = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore"):
        result = safe + np.log(np.exp(values - safe).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def log_partition(precision, information):
    sign, determinant = np.linalg.slogdet(precision)
    if np.any(sign <= 0):
        raise ValueError("Gaussian precision must be positive definite")
    mean = np.linalg.solve(precision, information[..., None])[..., 0]
    return 0.5 * (information.shape[-1] * np.log(2 * np.pi) - determinant
                  + np.sum(information * mean, axis=-1))


@dataclass(frozen=True)
class TrackEvidence:
    label: int
    first_detection: int
    end_scan: int
    times: tuple[int, ...]
    observations: tuple[tuple[float, ...] | None, ...]
    reached_existence: float

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(self.times))
        object.__setattr__(self, "observations", tuple(
            None if z is None else tuple(float(x) for x in z)
            for z in self.observations))
        if self.times != tuple(range(self.first_detection, self.end_scan + 1)):
            raise ValueError("History must include every scan, including misses")
        if len(self.observations) != len(self.times) or not self.times:
            raise ValueError("History length mismatch")
        if self.first_detection < 0 or self.observations[0] is None:
            raise ValueError("First scan must be a detection")

    @property
    def admissible(self):
        return (len(self.times) == 5 and self.reached_existence >= 0.9
                and sum(z is not None for z in self.observations) >= 2)


@dataclass(frozen=True)
class BirthLikelihood:
    label: int
    birth_scans: tuple[int, ...]
    precision: np.ndarray
    information_vector: np.ndarray
    log_scale: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "birth_scans", tuple(self.birth_scans))
        for name in ("precision", "information_vector", "log_scale"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        n = len(self.birth_scans)
        d = self.information_vector.shape[-1]
        if (self.precision.shape != (n, d, d)
                or self.information_vector.shape != (n, d)
                or self.log_scale.shape != (n,) or not n):
            raise ValueError("Invalid likelihood dimensions")
        if not (np.isfinite(self.precision).all()
                and np.isfinite(self.information_vector).all()
                and np.all(np.isfinite(self.log_scale) | np.isneginf(self.log_scale))):
            raise ValueError("Nonfinite likelihood")
        if not np.allclose(self.precision, self.precision.swapaxes(-1, -2)):
            raise ValueError("Likelihood precision is not symmetric")
        if np.min(np.linalg.eigvalsh(self.precision)) < -1e-8:
            raise ValueError("Likelihood precision is not positive semidefinite")

    def integrate_spread(self, covariance):
        """Integrate x ~ N(mu, V), retaining constants and singular factors."""
        inverse = np.linalg.inv(covariance)
        precision = inverse + self.precision
        solved_j = np.linalg.solve(precision, self.precision)
        solved_h = np.linalg.solve(precision, self.information_vector[..., None])[..., 0]
        j = self.precision - self.precision @ solved_j
        h = self.information_vector - (self.precision @ solved_h[..., None])[..., 0]
        c = (self.log_scale - 0.5 * np.linalg.slogdet(covariance)[1]
             - 0.5 * np.linalg.slogdet(precision)[1]
             + 0.5 * np.sum(self.information_vector * solved_h, axis=-1))
        return BirthLikelihood(self.label, self.birth_scans, (j + j.swapaxes(-1, -2)) / 2, h, c)

    def posterior(self, mean, covariance):
        inverse = np.linalg.inv(covariance)
        information = inverse @ mean
        j = inverse + self.precision
        h = information + self.information_vector
        cov = np.linalg.inv(j)
        means = (cov @ h[..., None])[..., 0]
        log_mass = (self.log_scale + log_partition(j, h)
                    - log_partition(inverse, information))
        return means, cov, np.exp(log_mass - logsumexp(log_mass))


def joint_observation_model(history, birth_scan, transition, process, observation, noise):
    """Y = A x_birth + epsilon with the full, correlated covariance."""
    detected = [(t, z) for t, z in zip(history.times, history.observations) if z is not None]
    if birth_scan > history.first_detection or birth_scan < 0:
        raise ValueError("Infeasible physical birth time")
    d = transition.shape[0]
    powers = [np.eye(d)]
    accumulated = [np.zeros((d, d))]
    for _ in range(history.end_scan - birth_scan):
        powers.append(transition @ powers[-1])
        accumulated.append(transition @ accumulated[-1] @ transition.T + process)
    design = np.concatenate([observation @ powers[t - birth_scan] for t, _ in detected])
    blocks = []
    for ti, _ in detected:
        row = []
        for tj, _ in detected:
            if ti >= tj:
                cross = powers[ti - tj] @ accumulated[tj - birth_scan]
            else:
                cross = accumulated[ti - birth_scan] @ powers[tj - ti].T
            row.append(observation @ cross @ observation.T + (noise if ti == tj else 0))
        blocks.append(row)
    covariance = np.block(blocks)
    values = np.concatenate([z for _, z in detected])
    return values, design, covariance


def history_likelihood(history, transition, process, observation, noise,
                       detection_probability=0.9, survival_probability=0.98,
                       birth_rate=0.125):
    """Likelihood at physical birth, including all feasible missed initial scans.

    The shared geometric survival model is conditional on survival through the
    history window. The simulator's fixed lifetime is an explicit mismatch.
    """
    if not 0 < detection_probability < 1 or not 0 < survival_probability <= 1 or birth_rate <= 0:
        raise ValueError("Invalid shared sensor/birth parameters")
    precisions, information, constants = [], [], []
    detections = sum(z is not None for z in history.observations)
    for birth in range(history.first_detection + 1):
        y, a, s = joint_observation_model(history, birth, transition, process, observation, noise)
        solved_a = np.linalg.solve(s, a)
        solved_y = np.linalg.solve(s, y)
        j = a.T @ solved_a
        h = a.T @ solved_y
        c = -0.5 * (y @ solved_y + np.linalg.slogdet(s)[1] + len(y) * np.log(2 * np.pi))
        elapsed = history.end_scan - birth
        c += (np.log(birth_rate) + elapsed * np.log(survival_probability)
              + detections * np.log(detection_probability)
              + (elapsed + 1 - detections) * np.log1p(-detection_probability))
        precisions.append((j + j.T) / 2)
        information.append(h)
        constants.append(c)
    return BirthLikelihood(history.label, tuple(range(history.first_detection + 1)),
                           precisions, information, constants)
