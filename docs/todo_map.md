# TODO Map

Current implementation status and next work items.

## Completed recently

- DP clutter model exists as a normalized density plus separate Poisson rate.
- LMB-style tracker accepts an optional clutter model.
- Association, existence updates, and birth decisions can use local clutter intensity.
- Fractional clutter responsibilities are fed back into adaptive DP clutter.
- Responsibility learning can be attenuated or gated.
- Delayed DP birth learning can defer birth-atom updates until a newborn track is confirmed.
- Fixed Gaussian-mixture clutter baseline added for hand-specified hotspot comparisons.
- Structured clutter metrics script writes CSV output for scalar, fixed-GMM, and adaptive-DP clutter.
- Plotting script generates paper-facing PDF summaries from the metrics CSV.
- Lightweight GitHub Actions workflow added for tests and smoke runs.

## Highest-value next items

1. Add a birth-rate model separated from DP birth density.
2. Reclustering for confirmed birth evidence instead of always appending a new atom.
3. Add a PMBM/GLMB posterior-predictive insertion interface sketch as code, not only docs.
4. Harden validation and add more shape/error tests.
5. Add harder structured-clutter scenarios: moving hotspot, unknown GMM baseline, nonstationary clutter rate.

## Paper-facing next items

- Run structured clutter metrics over harder scenarios.
- Plot hotspot-track-step proxy and total births for nonstationary clutter.
- Add the structured-clutter experiment subsection to the manuscript once scenario parameters are stable.
- Explain fixed GMM clutter as a hand-tuned/oracle-ish baseline.
- Verify bibliography and replace placeholders.
