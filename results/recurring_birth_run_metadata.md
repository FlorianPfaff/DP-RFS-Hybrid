# Recurring-birth development campaign provenance

This file describes the original seeds 0--99 campaign. Those results were
inspected while the method and reporting protocol were still being developed.
They are therefore development evidence, including the analyses that used
seeds 20--99, and are not the paper's locked headline evaluation.

- Scenario: 96 scans, two alternating recurring birth regions, 12 targets per
  realization, 14-scan target lifetimes, detection probability 0.9, and
  uniform Poisson clutter with mean 6 measurements per scan. The trackers use
  survival probability 0.98.
- Metric: RMS GOSPA with order 2, cutoff 10 m, and alpha 2. Localization,
  missed-target, and false-target powered components are accumulated per scan
  before taking the sequence-level root mean.
- Development seeds: paired seeds 0--99. The DP concentration was selected as
  10 from the candidates 1, 2, 5, 10, and 20.
- Native implementation: all six birth-model variants use identical truth,
  measurements, dynamics, sensing model, and greedy LMB-style outer tracker.
- PMBM reference: official Gaussian PMBM code from
  `Agarciafernandez/MTT` at
  `cc5c6df0e9343dc1345d91ed6ab95f01db4d1821`.
- Assignment dependency: `USNavalResearchLaboratory/TrackerComponentLibrary`
  at `593ce51d64cfb79fd4293968075e5de558c83338`.
- PMBM runtime: GNU Octave 9.2.0 with the statistics package in the
  `gnuoctave/octave:9.2.0` Apptainer image.

The native rows isolate the birth mechanism. The PMBM rows change the outer
filter as well and are therefore an algorithm-level reference, not a
single-factor ablation. See `recurring_birth_locked_run_metadata.md` for the
fresh seeds 100--199 evaluation used in paper claims.
