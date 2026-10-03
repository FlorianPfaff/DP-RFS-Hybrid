"""Collapsed DP/finite-mixture SMC and an exhaustive small-case reference."""

from dataclasses import dataclass

import numpy as np

from .coherent_evidence import log_partition, logsumexp


@dataclass(frozen=True)
class GaussianMixture:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def logpdf(self, values):
        values = np.atleast_2d(values)
        delta = values[:, None, :] - self.means
        inverse = np.linalg.inv(self.covariances)
        quadratic = np.einsum("nki,kij,nkj->nk", delta, inverse, delta)
        log_values = (np.log(self.weights) - 0.5 * (quadratic
                      + np.linalg.slogdet(self.covariances)[1]
                      + self.means.shape[-1] * np.log(2 * np.pi)))
        return logsumexp(log_values, axis=1)

    def project(self, matrix, noise):
        return GaussianMixture(self.weights, self.means @ matrix.T,
                               matrix @ self.covariances @ matrix.T + noise)


class CollapsedMixture:
    """Fully adapted proposals followed by invariant single-site Gibbs moves.

    Finite K uses a symmetric Dirichlet(alpha/K) distribution. Empty components
    are analytically combined, not treated as one component with alpha/K mass.
    """

    def __init__(self, mean, covariance, spread, alpha=1.0, components=None,
                 particles=128, sweeps=2, seed=0):
        if alpha <= 0 or particles < 1 or sweeps < 0 or (components is not None and components < 1):
            raise ValueError("Invalid mixture configuration")
        self.mean = np.asarray(mean)
        self.covariance = np.asarray(covariance)
        self.spread = np.asarray(spread)
        self.j0 = np.linalg.inv(covariance)
        self.h0 = self.j0 @ self.mean
        self.alpha, self.components = alpha, components
        self.particles, self.sweeps = particles, sweeps
        self.rng = np.random.default_rng(seed)
        self.sites = []
        self.labels = set()
        self.allocations = np.empty((particles, 0), dtype=int)
        self.times = np.empty((particles, 0), dtype=int)
        self.counts = np.zeros((particles, 1), dtype=int)
        self.j = np.zeros((particles, 1, len(mean), len(mean)))
        self.h = np.zeros((particles, 1, len(mean)))
        self.weights = np.full(particles, 1 / particles)
        self.resamples = 0

    def _options(self, site):
        occupied = self.counts > 0
        sizes = occupied.sum(axis=1)
        width = int(sizes.max()) + 1
        # Occupied slots first; exactly one representative empty slot per row.
        slots = np.argsort(~occupied, axis=1, kind="stable")[:, :width]
        valid = np.arange(width)[None, :] <= sizes[:, None]
        rows = np.arange(self.particles)[:, None]
        n = self.counts[rows, slots]
        empty = n == 0
        if self.components is None:
            mass = np.where(empty, self.alpha, n).astype(float)
        else:
            mass = np.where(empty, self.alpha * (self.components - sizes[:, None]) / self.components,
                            n + self.alpha / self.components)
        mass = np.where(valid, mass, 0)
        j = self.j0 + self.j[rows, slots]
        h = self.h0 + self.h[rows, slots]
        log_z = log_partition(j, h)
        posterior_j = j[:, :, None] + site.precision
        posterior_h = h[:, :, None] + site.information_vector
        with np.errstate(divide="ignore"):
            scores = (np.log(mass)[:, :, None] + site.log_scale
                      + log_partition(posterior_j, posterior_h) - log_z[:, :, None])
        return scores.reshape(self.particles, -1), slots

    def _sample(self, site):
        scores, slots = self._options(site)
        normalizer = logsumexp(scores, axis=1)
        probabilities = np.exp(scores - normalizer[:, None])
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        chosen = (cumulative < self.rng.random((self.particles, 1))).sum(axis=1)
        branch, time = np.divmod(chosen, len(site.birth_scans))
        return slots[np.arange(self.particles), branch], time, normalizer

    def _change(self, site, branches, times, sign):
        rows = np.arange(self.particles)
        self.counts[rows, branches] += sign
        self.j[rows, branches] += sign * site.precision[times]
        self.h[rows, branches] += sign * site.information_vector[times]
        empty = self.counts == 0
        self.j[empty] = 0
        self.h[empty] = 0

    def update(self, site):
        if site.label in self.labels:
            raise ValueError("Duplicate evidence label")
        self.labels.add(site.label)
        self.counts = np.pad(self.counts, ((0, 0), (0, 1)))
        self.j = np.pad(self.j, ((0, 0), (0, 1), (0, 0), (0, 0)))
        self.h = np.pad(self.h, ((0, 0), (0, 1), (0, 0)))
        branches, times, increment = self._sample(site)
        self._change(site, branches, times, 1)
        self.allocations = np.column_stack((self.allocations, branches))
        self.times = np.column_stack((self.times, times))
        self.sites.append(site)
        log_weights = np.log(self.weights) + increment
        self.weights = np.exp(log_weights - logsumexp(log_weights))
        if 1 / np.sum(self.weights ** 2) < self.particles / 2:
            positions = (self.rng.random() + np.arange(self.particles)) / self.particles
            indices = np.searchsorted(self.weights.cumsum(), positions).clip(max=self.particles - 1)
            for name in ("counts", "j", "h", "allocations", "times"):
                setattr(self, name, getattr(self, name)[indices].copy())
            self.weights.fill(1 / self.particles)
            self.resamples += 1
        for _ in range(self.sweeps):
            for i in self.rng.permutation(len(self.sites)):
                old_site = self.sites[i]
                self._change(old_site, self.allocations[:, i], self.times[:, i], -1)
                branches, times, _ = self._sample(old_site)
                self._change(old_site, branches, times, 1)
                self.allocations[:, i] = branches
                self.times[:, i] = times

    def predictive(self):
        weights, means, covariances = [], [], []
        total = len(self.sites) + self.alpha
        for p in range(self.particles):
            occupied = self.counts[p] > 0
            n = self.counts[p, occupied]
            j = self.j0 + self.j[p, occupied]
            cov = np.linalg.inv(j)
            mu = (cov @ (self.h0 + self.h[p, occupied])[..., None])[..., 0]
            mass = n if self.components is None else n + self.alpha / self.components
            residual = self.alpha if self.components is None else self.alpha * (1 - len(n) / self.components)
            weights.extend(self.weights[p] * mass / total)
            means.extend(mu)
            covariances.extend(cov + self.spread)
            if residual > 0:
                weights.append(self.weights[p] * residual / total)
                means.append(self.mean)
                covariances.append(self.covariance + self.spread)
        weights = np.asarray(weights)
        positive = weights > 0
        return GaussianMixture(weights[positive] / weights.sum(), np.asarray(means)[positive],
                               np.asarray(covariances)[positive])

    def probabilities(self):
        coassignment = np.einsum("p,pij->ij", self.weights,
                                self.allocations[:, :, None] == self.allocations[:, None, :])
        birth = [np.bincount(self.times[:, i], weights=self.weights, minlength=len(s.birth_scans))
                 for i, s in enumerate(self.sites)]
        return coassignment, birth

    @property
    def model_size(self):
        return float(self.weights @ (self.counts > 0).sum(axis=1))


def exact_posterior(sites, mean, covariance, spread, alpha=1.0, components=None):
    """Enumerate restricted-growth partitions and all time assignments (n <= 5).

    Independent scalar recursion, deliberately not using the SMC proposal code.
    """
    if len(sites) > 5 or len({s.label for s in sites}) != len(sites):
        raise ValueError("Exact reference requires <=5 unique histories")
    j0 = np.linalg.inv(covariance)
    h0 = j0 @ mean
    records = []

    def visit(i, clusters, allocations, times, log_weight):
        if i == len(sites):
            records.append((log_weight, clusters, allocations, times))
            return
        site = sites[i]
        for k in range(len(clusters) + 1):
            is_new = k == len(clusters)
            if is_new:
                mass = alpha if components is None else alpha * (components - k) / components
                n, j, h = 0, j0, h0
            else:
                n, j, h = clusters[k]
                mass = n if components is None else n + alpha / components
            if mass <= 0:
                continue
            for b in range(len(site.birth_scans)):
                new_j, new_h = j + site.precision[b], h + site.information_vector[b]
                increment = (np.log(mass / (i + alpha)) + site.log_scale[b]
                             + log_partition(new_j, new_h) - log_partition(j, h))
                updated = list(clusters)
                cluster = (n + 1, new_j, new_h)
                if is_new:
                    updated.append(cluster)
                else:
                    updated[k] = cluster
                visit(i + 1, updated, allocations + [k], times + [b], log_weight + increment)

    visit(0, [], [], [], 0.0)
    log_weights = np.array([r[0] for r in records])
    weights = np.exp(log_weights - logsumexp(log_weights))
    coassignment = np.zeros((len(sites), len(sites)))
    birth = [np.zeros(len(s.birth_scans)) for s in sites]
    pred_weights, means, covariances = [], [], []
    for weight, (_, clusters, allocations, times) in zip(weights, records):
        allocations = np.array(allocations)
        coassignment += weight * (allocations[:, None] == allocations[None, :])
        for i, t in enumerate(times):
            birth[i][t] += weight
        total = len(sites) + alpha
        for n, j, h in clusters:
            cov = np.linalg.inv(j)
            pred_weights.append(weight * (n if components is None else n + alpha / components) / total)
            means.append(cov @ h)
            covariances.append(cov + spread)
        residual = alpha if components is None else alpha * (1 - len(clusters) / components)
        if residual > 0:
            pred_weights.append(weight * residual / total)
            means.append(mean)
            covariances.append(covariance + spread)
    return {"coassignment": coassignment, "birth_time": birth, "states": len(records),
            "predictive": GaussianMixture(np.array(pred_weights), np.array(means), np.array(covariances))}
