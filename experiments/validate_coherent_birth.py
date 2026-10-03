"""Independent-run SMC accuracy gate. Run on the configured compute server."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from dp_rfs_hybrid.coherent_evidence import BirthLikelihood, TrackEvidence, history_likelihood
from dp_rfs_hybrid.coherent_mixture import CollapsedMixture, exact_posterior
from dp_rfs_hybrid.coherent_study import TRANSITION, PROCESS, OBSERVATION, NOISE, SPREAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    sites = [BirthLikelihood(i, (0, 1), np.array([[[2.]], [[1.]]]),
                            np.array([[2*x], [x-.2]]), np.array([-x*x, -.5*(x-.2)**2]) - np.log(2))
             for i, x in enumerate([-.5, .2, 2.5, 2.8, .1])]
    prior = (np.zeros(1), np.eye(1)*4, np.eye(1)*.3)
    rows = []
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    grid = np.linspace(-12, 12, 241)[:, None]
    for components, alpha in [(None, .5), (None, 5.), (1, 1.), (4, 1.)]:
        reference = exact_posterior(sites, *prior, alpha=alpha, components=components)
        exact_prob = np.concatenate((reference["coassignment"].ravel(), np.array(reference["birth_time"]).ravel()))
        exact_density = np.exp(reference["predictive"].logpdf(grid))
        for particles in [64, 128, 256]:
            probabilities, densities = [], []
            for seed in range(args.runs):
                learner = CollapsedMixture(*prior, alpha=alpha, components=components, particles=particles, seed=seed)
                for site in sites:
                    learner.update(site)
                coassignment, birth = learner.probabilities()
                probabilities.append(np.concatenate((coassignment.ravel(), np.array(birth).ravel())))
                densities.append(np.exp(learner.predictive().logpdf(grid)))
            error = np.asarray(probabilities) - exact_prob
            rows.append({"components": components, "alpha": alpha, "particles": particles,
                         "exact_states": reference["states"], "runs": args.runs,
                         "max_mean_probability_error": float(np.abs(error.mean(axis=0)).max()),
                         "mean_max_absolute_run_error": float(np.abs(error).max(axis=1).mean()),
                         "predictive_mean_integrated_absolute_error": float(integrate(np.abs(np.mean(densities, axis=0)-exact_density), grid[:, 0])),
                         "probabilities": np.asarray(probabilities).tolist(),
                         "reference_probabilities": exact_prob.tolist()})
            print({k: v for k, v in rows[-1].items() if "probabilities" not in k}, flush=True)
    trajectory_sites = []
    for i, x in enumerate([-.5, .2, 2.5, 2.8, .1]):
        history = TrackEvidence(i, 1, 5, tuple(range(1, 6)),
                                tuple((x+t, .3*t) if t != 3 else None for t in range(1, 6)), .99)
        trajectory_sites.append(history_likelihood(history, TRANSITION, PROCESS, OBSERVATION, NOISE).integrate_spread(SPREAD))
    trajectory_prior = (np.array([0., 0., 1., .3]), np.diag([4., 4., .5, .5]), SPREAD)
    for components in [None, 4]:
        reference = exact_posterior(trajectory_sites, *trajectory_prior, components=components)
        exact_prob = np.concatenate((reference["coassignment"].ravel(), np.array(reference["birth_time"]).ravel()))
        probabilities = []
        for seed in range(args.runs):
            learner = CollapsedMixture(*trajectory_prior, components=components, seed=seed)
            for site in trajectory_sites:
                learner.update(site)
            coassignment, birth = learner.probabilities()
            probabilities.append(np.concatenate((coassignment.ravel(), np.array(birth).ravel())))
        error = np.asarray(probabilities)-exact_prob
        rows.append({"case": "four_dimensional_histories", "components": components, "alpha": 1., "particles": 128,
                     "exact_states": reference["states"], "runs": args.runs,
                     "max_mean_probability_error": float(np.abs(error.mean(axis=0)).max()),
                     "probabilities": np.asarray(probabilities).tolist(), "reference_probabilities": exact_prob.tolist()})
        print({k:v for k,v in rows[-1].items() if "probabilities" not in k}, flush=True)
    result = {"rows": rows, "passed": args.runs >= 64 and all(r["max_mean_probability_error"] < .03 for r in rows if r["particles"] == 128),
              "seconds": time.perf_counter()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("Particle correctness gate failed")


if __name__ == "__main__":
    main()
