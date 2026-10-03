"""Controlled, separately versioned birth-learning study; legacy code unchanged."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import time

import numpy as np

from .coherent_evidence import BirthLikelihood, TrackEvidence, history_likelihood, logsumexp
from .coherent_mixture import CollapsedMixture, GaussianMixture
from .dp_birth import BirthDecision
from .gaussian import GaussianState
from .lmb_tracker import LabeledMultiBernoulliTracker
from .metrics import gospa, _linear_sum_assignment


CONDITIONS = ("stationary2", "stationary4", "stationary8", "nonrecurring", "high_clutter", "drift")
ALPHAS = (.5, 1, 2, 5, 10, 20, 50, 100, 200)
BASE_MEAN = np.zeros(4)
BASE_COV = np.diag([30.**2, 22.**2, 2.**2, 2.**2])
SPREAD = np.diag([.55**2, .55**2, .08**2, .08**2])
TRANSITION = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
GAIN = np.array([[.5, 0.], [0., .5], [1., 0.], [0., 1.]])
PROCESS = .06**2 * GAIN @ GAIN.T + np.eye(4)*1e-6
OBSERVATION = np.eye(4)[:2]
NOISE = np.eye(2)*.6**2


@dataclass(frozen=True)
class Sensor:
    detection: float = .9
    clutter: float = 6.
    survival: float = .98
    birth_rate: float = .125


def sensor_for(condition):
    return Sensor(detection=.7, clutter=12.) if condition == "high_clutter" else Sensor()


def configurations():
    result = [{"learner": "dp", "alpha": a} for a in ALPHAS]
    result += [{"learner": "finite", "alpha": a, "components": k} for a in ALPHAS for k in (1, 2, 4, 8, 16)]
    result += [{"learner": "kde", "bandwidth": h, "residual": r} for h in (.5, 1., 2.) for r in (.05, .2, .5)]
    return result


def configuration_id(config):
    return "_".join(f"{key}-{value}" for key, value in sorted(config.items()))


def scenario(seed, condition):
    if condition not in CONDITIONS or seed >= 300:
        raise ValueError("Unknown condition or reserved seed")
    location, dynamics, detection, clutter = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    sensor = sensor_for(condition)
    number = int(condition[-1]) if condition.startswith("stationary") else 2
    centers = np.array([[-24., -10., 1.15, .35], [22., 12., -1., -.3],
                        [-22., 13., 1., -.4], [24., -12., -1.1, .4],
                        [0., -22., .2, 1.], [0., 22., -.2, -1.],
                        [-35., 0., 1.2, .1], [35., 0., -1.2, -.1]])
    truth = np.full((16, 128, 4), np.nan)
    births = np.arange(0, 128, 8)
    regions = []
    for i, birth in enumerate(births):
        region = int(location.integers(number))
        center = centers[region].copy()
        if condition == "nonrecurring":
            angle = 2*np.pi*i/16
            center = np.array([30*np.cos(angle), 22*np.sin(angle), -np.cos(angle), -.7*np.sin(angle)])
            region = i
        elif condition == "drift":
            center[:2] += np.array([8., 0.]) * birth / 127
        regions.append(region)
        state = location.multivariate_normal(center, SPREAD)
        for t in range(birth, min(128, birth + 14)):
            if t > birth:
                state = TRANSITION @ state + dynamics.multivariate_normal(np.zeros(4), PROCESS)
            truth[i, t] = state
    measurements, sources = [], []
    for t in range(128):
        z, ids = [], []
        for i in range(16):
            if np.isfinite(truth[i, t, 0]) and detection.random() < sensor.detection:
                z.append(truth[i, t, :2] + detection.multivariate_normal(np.zeros(2), NOISE))
                ids.append(i)
        count = clutter.poisson(sensor.clutter)
        z.extend(clutter.uniform([-50., -35.], [50., 35.], (count, 2)))
        ids.extend([-1]*count)
        measurements.append(np.asarray(z).reshape(-1, 2))
        sources.append(np.array(ids))
    return {"truth": truth, "births": births, "regions": regions,
            "measurements": measurements, "sources": sources, "sensor": sensor}


class ReferenceKDE:
    def __init__(self, bandwidth=1., residual=.2, fixed=False):
        self.bandwidth, self.residual, self.fixed = bandwidth, residual, fixed
        self.kernels, self.labels = [], set()

    def update(self, raw):
        if raw.label in self.labels:
            raise ValueError("Duplicate evidence label")
        self.labels.add(raw.label)
        means, covariance, mass = raw.posterior(BASE_MEAN, BASE_COV + SPREAD)
        bandwidth = np.diag([self.bandwidth**2]*2 + [(self.bandwidth/10)**2]*2)
        self.kernels.append(GaussianMixture(mass, means, covariance + bandwidth))

    def predictive(self):
        if self.fixed or not self.kernels:
            return GaussianMixture(np.ones(1), BASE_MEAN[None], (BASE_COV + SPREAD)[None])
        weights = np.concatenate(([self.residual], *[(1-self.residual)*k.weights/len(self.kernels) for k in self.kernels]))
        means = np.concatenate((BASE_MEAN[None], *[k.means for k in self.kernels]))
        covs = np.concatenate(((BASE_COV+SPREAD)[None], *[k.covariances for k in self.kernels]))
        return GaussianMixture(weights, means, covs)

    @property
    def model_size(self):
        return len(self.kernels)


class BirthAdapter:
    """Identical intensity, admission, and mixture moment initialization."""
    birth_probability = .35
    odds_threshold = .01

    def __init__(self, config, sensor, seed=0, particles=128):
        self.config, self.sensor = config, sensor
        self.clutter_intensity = sensor.clutter / 7000
        if config["learner"] in ("dp", "finite"):
            self.learner = CollapsedMixture(BASE_MEAN, BASE_COV, SPREAD,
                                           alpha=config["alpha"], components=config.get("components"),
                                           seed=seed, particles=particles)
        else:
            self.learner = ReferenceKDE(config.get("bandwidth", 1.), config.get("residual", .2),
                                        fixed=config["learner"] == "fixed")
        self.refresh()

    def refresh(self):
        self.density = self.learner.predictive()
        projected = self.density.project(OBSERVATION, NOISE)
        self.projected = projected
        self.inverse = np.linalg.inv(projected.covariances)
        self.log_constants = np.log(projected.weights) - .5*(np.linalg.slogdet(projected.covariances)[1] + 2*np.log(2*np.pi))
        self.gains = self.density.covariances @ OBSERVATION.T @ self.inverse
        self.posterior_covs = self.density.covariances - self.gains @ projected.covariances @ self.gains.swapaxes(-1, -2)

    def process(self, measurement, clutter_intensity=None):
        delta = measurement - self.projected.means
        log_weights = self.log_constants - .5*np.einsum("ki,kij,kj->k", delta, self.inverse, delta)
        log_density = logsumexp(log_weights)
        density = float(np.exp(log_density))
        intensity = self.sensor.birth_rate * density
        kappa = self.clutter_intensity if clutter_intensity is None else clutter_intensity
        odds = intensity / kappa
        state = None
        if odds > self.odds_threshold:
            weights = np.exp(log_weights-log_density)
            means = self.density.means + (self.gains @ delta[..., None])[..., 0]
            mean = weights @ means
            offsets = means-mean
            covariance = np.einsum("k,kij->ij", weights, self.posterior_covs) + np.einsum("k,ki,kj->ij", weights, offsets, offsets)
            state = GaussianState(mean, (covariance+covariance.T)/2)
        return BirthDecision(state is not None, "mixture", float(odds), state=state,
                             clutter_intensity=kappa, birth_density=density, birth_intensity=intensity,
                             existence_probability=self.birth_probability)

    def decay_counts(self):
        pass

    def learn(self, history, initial_prior=None):
        raw = history_likelihood(history, TRANSITION, PROCESS, OBSERVATION, NOISE,
                                  self.sensor.detection, self.sensor.survival, self.sensor.birth_rate)
        if isinstance(self.learner, CollapsedMixture):
            site = raw.integrate_spread(SPREAD)
            ablation = self.config.get("ablation")
            if ablation == "prior_feedback":
                # Deliberate error control: reuse the initialization density as
                # an extra likelihood, rather than dividing it out.
                mean, cov = initial_prior
                inverse = np.linalg.inv(cov)
                info = inverse @ mean
                scale = -.5*(mean @ info + np.linalg.slogdet(cov)[1] + 4*np.log(2*np.pi))
                site = BirthLikelihood(site.label, site.birth_scans, site.precision + inverse,
                                       site.information_vector + info, site.log_scale + scale)
            elif ablation == "confirmation_time":
                transforms = np.array([np.linalg.matrix_power(TRANSITION, -(history.end_scan-b)) for b in site.birth_scans])
                site = BirthLikelihood(site.label, (history.end_scan,)*len(site.birth_scans),
                                       transforms.swapaxes(-1, -2) @ site.precision @ transforms,
                                       (transforms.swapaxes(-1, -2) @ site.information_vector[..., None])[..., 0],
                                       site.log_scale)
            self.learner.update(site)
        else:
            self.learner.update(raw)
        self.refresh()


class HistoryTracker(LabeledMultiBernoulliTracker):
    def __init__(self, adapter):
        super().__init__(TRANSITION, PROCESS, OBSERVATION, NOISE, adapter,
                         survival_probability=adapter.sensor.survival,
                         detection_probability=adapter.sensor.detection, association_threshold=.2)
        self.scan = -1
        self.pending = {}
        self.admitted = []
        self.closed = []
        self.total_initiations = 0

    def _process_birth_measurement(self, measurement, clutter_intensity):
        decision = super()._process_birth_measurement(measurement, clutter_intensity)
        if decision.accepted:
            density = self.birth_model.density
            mean = density.weights @ density.means
            offsets = density.means-mean
            cov = (np.einsum("k,kij->ij", density.weights, density.covariances)
                   + np.einsum("k,ki,kj->ij", density.weights, offsets, offsets))
            self.new_observations.append((tuple(measurement), (mean, cov)))
        return decision

    def step(self, measurements, learn=True):
        self.scan += 1
        self.new_observations = []
        summary = super().step(measurements)
        observed = {label: tuple(measurements[index]) for label, index in summary.assignments}
        for label, (z, prior) in zip(summary.births, self.new_observations):
            self.pending[label] = {"first": self.scan, "values": [], "maximum": 0., "prior": prior}
            observed[label] = z
        self.total_initiations += len(summary.births)
        existence = {track.label: track.existence for track in self.tracks}
        self.closed = []
        for label, record in list(self.pending.items()):
            record["values"].append(observed.get(label))
            record["maximum"] = max(record["maximum"], existence.get(label, 0.))
            if self.scan == record["first"] + 4:
                evidence = TrackEvidence(label, record["first"], self.scan,
                                         tuple(range(record["first"], self.scan+1)), tuple(record["values"]), record["maximum"])
                if evidence.admissible:
                    self.closed.append((evidence, record["prior"]))
                    self.admitted.append(evidence)
                del self.pending[label]
        # All associations, admissions, and estimates for this scan are fixed.
        if learn:
            for evidence, prior in self.closed:
                self.birth_model.learn(evidence, prior)
        return summary


@lru_cache(maxsize=8)
def replay_stream(seed, condition):
    data = scenario(seed, condition)
    tracker = HistoryTracker(BirthAdapter({"learner": "fixed"}, data["sensor"]))
    stream = []
    for measurements in data["measurements"]:
        tracker.step(measurements, learn=False)
        stream.extend(tracker.closed)
    return data, stream


def run_trial(seed, condition, config, mode="closed", particles=128, algorithm_seed=0):
    started = time.perf_counter()
    data, stream = replay_stream(seed, condition) if mode == "replay" else (scenario(seed, condition), [])
    # Separate algorithm randomness, with matching particle random streams for DP/finite.
    random_seed = np.random.SeedSequence([seed, CONDITIONS.index(condition), algorithm_seed, 9981])
    adapter = BirthAdapter(config, data["sensor"], random_seed, particles)
    tracker = HistoryTracker(adapter)
    by_scan = {}
    for history, prior in stream:
        by_scan.setdefault(history.end_scan, []).append((history, prior))
    loss, distances, missed, false = [], [], [], []
    target_labels = [set() for _ in range(16)]
    first_confirmed = np.full(16, np.nan)
    history_diagnostics = []
    for t in range(128):
        if t % 8 == 0:
            loss.append(float(-adapter.density.logpdf(data["truth"][t//8, t])[0]))
        if mode == "replay":
            closed = by_scan.get(t, [])
            for history, prior in closed:
                adapter.learn(history, prior)
        else:
            tracker.step(data["measurements"][t])
            closed = tracker.closed
            estimates = tracker.estimates()
            truth_ids = np.flatnonzero(np.isfinite(data["truth"][:, t, 0]))
            truth = data["truth"][truth_ids, t, :2]
            points = np.array([track.state.mean[:2] for track in estimates]).reshape(-1, 2)
            metric = gospa(truth, points)
            distances.append(metric.powered_distance)
            missed.append(metric.missed_count)
            false.append(metric.false_count)
            if len(truth) and len(points):
                costs = np.linalg.norm(truth[:, None]-points, axis=-1)
                for i, j in _linear_sum_assignment(costs):
                    if costs[i, j] < 10:
                        target = truth_ids[i]
                        target_labels[target].add(estimates[j].label)
                        if estimates[j].existence >= .9 and np.isnan(first_confirmed[target]):
                            first_confirmed[target] = t
        for history, _ in closed:
            # Diagnostic truth attribution only; never used for admission/learning.
            ids = []
            for scan, z in zip(history.times, history.observations):
                if z is not None:
                    match = np.flatnonzero(np.all(data["measurements"][scan] == np.asarray(z), axis=1))
                    if len(match):
                        ids.append(int(data["sources"][scan][match[0]]))
            true_ids = [i for i in ids if i >= 0]
            target = max(set(true_ids), key=true_ids.count) if true_ids else -1
            purity = ids.count(target)/len(ids) if ids else 0.
            birth_estimate = None
            if isinstance(adapter.learner, CollapsedMixture):
                index = next(i for i, site in enumerate(adapter.learner.sites) if site.label == history.label)
                probabilities = adapter.learner.probabilities()[1][index]
                birth_estimate = float(probabilities @ np.array(adapter.learner.sites[index].birth_scans))
            history_diagnostics.append({"label": history.label, "first": history.first_detection,
                                         "end": history.end_scan, "target": target, "purity": purity,
                                         "birth_estimate": birth_estimate,
                                         "true_birth": int(data["births"][target]) if target >= 0 else None})
    latencies = [float(first_confirmed[i]-data["births"][i]) for i in range(16) if np.isfinite(first_confirmed[i])]
    errors = [abs(h["birth_estimate"]-h["true_birth"]) for h in history_diagnostics
              if h["birth_estimate"] is not None and h["true_birth"] is not None]
    return {"seed": seed, "condition": condition, "config": config, "mode": mode,
            "particles": particles, "algorithm_seed": algorithm_seed, "status": "ok",
            "gospa": float(np.sqrt(np.mean(distances))) if distances else None,
            "log_loss": float(np.mean(loss)), "birth_losses": loss,
            "missed": float(np.mean(missed)) if missed else None, "false": float(np.mean(false)) if false else None,
            "confirmation_delay": float(np.mean(latencies)) if latencies else None,
            "unconfirmed_targets": int(np.isnan(first_confirmed).sum()) if mode != "replay" else None,
            "fragmentation": float(np.mean([max(0, len(x)-1) for x in target_labels])) if mode != "replay" else None,
            "birth_time_error": float(np.mean(errors)) if errors else None,
            "model_size": adapter.learner.model_size, "histories": history_diagnostics,
            "admitted": len(history_diagnostics), "initiations": tracker.total_initiations if mode != "replay" else None,
            "runtime": time.perf_counter()-started, "sensor": asdict(data["sensor"])}
