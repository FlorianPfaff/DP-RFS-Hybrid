# TODO Map

Current implementation status and next work items.

## Completed recently

- DP clutter model exists as a normalized density plus separate Poisson rate.
- LMB-style tracker accepts an optional clutter model.
- Association, existence updates, and birth decisions can use local clutter intensity.
- Fractional clutter responsibilities are fed back into adaptive DP clutter.
- Responsibility learning can be attenuated or gated.
- Delayed-confirmation DP birth learning is available as an opt-in tracker mode.
- Fixed Gaussian-mixture clutter baseline added for hand-specified hotspot comparisons.
- Structured clutter metrics script writes CSV output for scalar, fixed-GMM, and adaptive-DP clutter.
- Plotting script generates paper-facing PDF summaries from the metrics CSV.
- Lightweight GitHub Actions workflow added for tests and smoke runs.

## Highest-value next items

1. Wait for hosted CI on the delayed-birth commits and fix any failures.
2. Run the metrics script and inspect the scalar/GMM/DP ordering after delayed birth learning is enabled in an experiment.
3. Add a PMBM/GLMB posterior-predictive insertion interface sketch.
4. Add a fixed Gaussian-mixture baseline to the reusable experiment helper, not only the metrics script.
5. Harden validation and add more shape/error tests.

## Paper-facing next items

- Add delayed-birth learning to the two-time-scale learning discussion.
- Run structured clutter metrics over enough seeds.
- Plot hotspot-track-step proxy and total births.
- Add the structured-clutter experiment subsection to the manuscript.
- Explain fixed GMM clutter as a hand-tuned/oracle-ish baseline.
- Verify bibliography and replace placeholders.
