# TODO Map

Current implementation status and next work items.

## Completed recently

- DP clutter model exists as a normalized density plus separate Poisson rate.
- LMB-style tracker accepts an optional DP clutter model.
- Association, existence updates, and birth decisions can use local DP clutter intensity.
- Fractional clutter responsibilities are fed back into the DP clutter model.
- Responsibility learning can be attenuated or gated.
- Structured clutter metrics script writes CSV output for fixed scalar clutter vs adaptive DP clutter.

## Highest-value next items

1. Add delayed-confirmation DP birth learning.
2. Add a plotting script for `results/structured_clutter_metrics.csv`.
3. Add a fixed Gaussian-mixture clutter baseline.
4. Add a lightweight CI workflow.
5. Add a PMBM/GLMB posterior-predictive insertion interface sketch.

## Paper-facing next items

- Run structured clutter metrics over enough seeds.
- Plot hotspot-track-step proxy and total births.
- Add the structured-clutter experiment subsection to the manuscript.
- Verify bibliography and replace placeholders.
