import numpy as np
import pytest

from dp_rfs_hybrid.coherent_evidence import (
    BirthLikelihood, TrackEvidence, history_likelihood, joint_observation_model,
    log_partition, logsumexp,
)
from dp_rfs_hybrid.coherent_mixture import CollapsedMixture, exact_posterior


def history(label=1):
    return TrackEvidence(label, 2, 6, tuple(range(2, 7)),
                         ((2.2,), None, (4.1,), (5.2,), (6.0,)), 0.99)


def model():
    return (np.array([[1., 1.], [0., 1.]]), np.diag([0.04, 0.02]),
            np.array([[1., 0.]]), np.array([[0.3]]))


def test_immutable_history_and_all_birth_times():
    evidence = history()
    assert evidence.admissible
    likelihood = history_likelihood(evidence, *model())
    assert likelihood.birth_scans == (0, 1, 2)
    with pytest.raises(ValueError):
        likelihood.precision[0, 0, 0] = 1
    with pytest.raises(ValueError):
        TrackEvidence(1, 1, 3, (1, 3), ((1.,), (2.,)), .99)


def test_joint_likelihood_matches_kalman_and_rts():
    evidence = history()
    f, q, h, r = model()
    site = history_likelihood(evidence, f, q, h, r)
    m0, p0 = np.array([0., 1.]), np.diag([4., .3])
    inverse = np.linalg.inv(p0)
    posterior_means, posterior_covs, _ = site.posterior(m0, p0)
    for b in site.birth_scans:
        m, p = m0.copy(), p0.copy()
        filtered, predicted = [], []
        log_likelihood = 0.
        for t in range(b, evidence.end_scan + 1):
            if t > b:
                m, p = f @ m, f @ p @ f.T + q
            predicted.append((m.copy(), p.copy()))
            z = evidence.observations[t - evidence.first_detection] if t >= evidence.first_detection else None
            if z is not None:
                v, s = np.array(z) - h @ m, h @ p @ h.T + r
                log_likelihood += -.5 * (v @ np.linalg.solve(s, v) + np.linalg.slogdet(s)[1] + np.log(2 * np.pi))
                gain = np.linalg.solve(s, h @ p).T
                m, p = m + gain @ v, p - gain @ s @ gain.T
            filtered.append((m.copy(), p.copy()))
        detections = 4
        log_likelihood += (np.log(.125) + (6 - b) * np.log(.98)
                           + detections * np.log(.9) + (7 - b - detections) * np.log(.1))
        canonical = (site.log_scale[b] + log_partition(inverse + site.precision[b], inverse @ m0 + site.information_vector[b])
                     - log_partition(inverse, inverse @ m0))
        np.testing.assert_allclose(canonical, log_likelihood, atol=1e-10)
        for k in range(len(filtered) - 2, -1, -1):
            mf, pf = filtered[k]
            mp, pp = predicted[k + 1]
            gain = np.linalg.solve(pp, f @ pf).T
            m, p = mf + gain @ (m - mp), pf + gain @ (p - pp) @ gain.T
        np.testing.assert_allclose(posterior_means[b], m, atol=1e-10)
        np.testing.assert_allclose(posterior_covs[b], p, atol=1e-10)
        y, a, s = joint_observation_model(evidence, b, f, q, h, r)
        assert s[0, -1] > 0 if b < 2 else s[1, -1] > 0
        joint_cov = s + a @ p0 @ a.T
        joint_delta = y - a @ m0
        spatial = -.5 * (joint_delta @ np.linalg.solve(joint_cov, joint_delta)
                         + np.linalg.slogdet(joint_cov)[1] + len(y) * np.log(2 * np.pi))
        np.testing.assert_allclose(spatial, log_likelihood - (np.log(.125) + (6 - b) * np.log(.98)
                                   + 4 * np.log(.9) + (3 - b) * np.log(.1)), atol=1e-10)


def test_prior_free_and_identical_posteriors_do_not_hide_likelihood():
    # q=N(1,.5) results both from pi=N(0,1), y=2 and pi=N(2,1), y=0.
    f = h = r = np.ones((1, 1))
    q = np.zeros((1, 1))
    sites = [history_likelihood(TrackEvidence(i, 0, 0, (0,), ((y,),), .99), f, q, h, r)
             for i, y in enumerate([2., 0.])]
    a = sites[0].posterior(np.array([0.]), np.eye(1))[0]
    b = sites[1].posterior(np.array([2.]), np.eye(1))[0]
    np.testing.assert_allclose(a, b)
    assert not np.allclose(sites[0].information_vector, sites[1].information_vector)
    for prior in [0., 100.]:
        sites[0].posterior(np.array([prior]), np.eye(1))
        again = history_likelihood(TrackEvidence(0, 0, 0, (0,), ((2.,),), .99), f, q, h, r)
        np.testing.assert_array_equal(again.information_vector, sites[0].information_vector)


def test_convolution_matches_direct_joint_and_rank_deficiency():
    f, q, h, r = model()
    evidence = TrackEvidence(1, 0, 0, (0,), ((2.,),), .99)
    raw = history_likelihood(evidence, f, q, h, r)
    assert np.linalg.matrix_rank(raw.precision[0]) == 1
    v = np.diag([.4, .1])
    integrated = raw.integrate_spread(v)
    mu = np.array([1., 3.])
    value = integrated.log_scale[0] + integrated.information_vector[0] @ mu - .5 * mu @ integrated.precision[0] @ mu
    expected = -.5 * (1 / .7 + np.log(2 * np.pi * .7)) + np.log(.125 * .9)
    np.testing.assert_allclose(value, expected)
    assert np.linalg.eigvalsh(integrated.precision).min() >= -1e-12


def small_sites():
    return [BirthLikelihood(i, (0, 1), np.array([[[2.]], [[1.]]]),
                            np.array([[2 * x], [x - .2]]), np.array([-x*x, -.5*(x-.2)**2]) - np.log(2))
            for i, x in enumerate([-.5, .2, 2.5, 2.8, .1])]


@pytest.mark.parametrize("components", [None, 1, 4])
def test_smc_exact_reference_and_normalization(components):
    args = (np.zeros(1), np.eye(1) * 4, np.eye(1) * .3)
    sites = small_sites()[:3]
    exact = exact_posterior(sites, *args, components=components)
    learner = CollapsedMixture(*args, components=components, particles=512, seed=42)
    for site in sites:
        learner.update(site)
    actual = learner.probabilities()
    np.testing.assert_allclose(actual[0], exact["coassignment"], atol=.1)
    np.testing.assert_allclose(actual[1], exact["birth_time"], atol=.1)
    predictive = learner.predictive()
    np.testing.assert_allclose(predictive.weights.sum(), 1.)
    assert np.linalg.eigvalsh(predictive.covariances).min() > 0
    grid = np.linspace(-30, 30, 2001)[:, None]
    integral = np.trapz(np.exp(predictive.logpdf(grid)), grid[:, 0])
    np.testing.assert_allclose(integral, 1., atol=1e-7)
    with pytest.raises(ValueError, match="Duplicate"):
        learner.update(sites[0])


def test_empty_constant_likelihood_and_low_information():
    args = (np.zeros(1), np.eye(1) * 4, np.eye(1) * .3)
    learner = CollapsedMixture(*args)
    before = learner.predictive().logpdf([[-2.], [0.], [1.]])
    constant = BirthLikelihood(99, (0, 1), np.zeros((2, 1, 1)), np.zeros((2, 1)), np.log([.3, .7]))
    learner.update(constant)
    np.testing.assert_allclose(learner.predictive().logpdf([[-2.], [0.], [1.]]), before)
    exact = exact_posterior(small_sites()[:2] + [constant], *args)
    without = exact_posterior(small_sites()[:2], *args)
    # Use a fresh label: neutrality also holds after informative observations.
    assert np.allclose(learner.probabilities()[1][0].sum(), 1.)
    np.testing.assert_allclose(exact["predictive"].logpdf([[0.], [2.]]),
                               without["predictive"].logpdf([[0.], [2.]]), atol=1e-10)


def test_deterministic_replay():
    args = (np.zeros(1), np.eye(1), np.eye(1))
    a, b = CollapsedMixture(*args, seed=8), CollapsedMixture(*args, seed=8)
    for site in small_sites():
        a.update(site)
        b.update(site)
    np.testing.assert_array_equal(a.allocations, b.allocations)
    np.testing.assert_array_equal(a.times, b.times)
