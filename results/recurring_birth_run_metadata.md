# Recurring-birth result provenance

- Scenario: 96 scans, two alternating recurring birth regions, 12 targets per
  realization, 14-scan target lifetimes, detection probability 0.9, and
  uniform Poisson clutter with mean 6 measurements per scan. The trackers use
  survival probability 0.98.
- Metric: RMS GOSPA with order 2, cutoff 10 m, and alpha 2. Localization,
  missed-target, and false-target powered components are accumulated per scan
  before taking the sequence-level root mean.
- Development set: paired seeds 0--19. The DP concentration was selected as
  10 from the candidates 1, 2, 5, 10, and 20.
- Evaluation set: paired seeds 20--99. Headline means, confidence intervals,
  figures, and pairwise tests use only these 80 held-out realizations.
- Native implementation: all six birth-model variants use identical truth,
  measurements, dynamics, sensing model, and greedy LMB-style outer tracker.
- PMBM reference: official Gaussian PMBM code from
  `Agarciafernandez/MTT` at
  `cc5c6df0e9343dc1345d91ed6ab95f01db4d1821`.
- Assignment dependency: `USNavalResearchLaboratory/TrackerComponentLibrary`
  at `593ce51d64cfb79fd4293968075e5de558c83338`.
- PMBM runtime: GNU Octave 9.2.0 with the statistics package in the
  `gnuoctave/octave:9.2.0` Apptainer image.
- Pairwise uncertainty: 20,000 paired bootstrap resamples of the mean
  sequence-level RMS GOSPA difference.
- Pairwise test: 50,000 random paired sign flips; the smallest reportable
  Monte Carlo p-value is approximately `2e-5`.

The native rows isolate the birth mechanism. The PMBM rows change the outer
filter as well and are therefore an algorithm-level reference, not a
single-factor ablation. The oracle rows use the true two-component birth
mixture and quantify the remaining gap to known birth structure.

The recurrence ablation uses the held-out seeds and varies the spawn interval
over 6, 8, 12, and 16 scans, corresponding to 16, 12, 8, and 6 target birth
events. It holds all tracker parameters fixed. This tests the expected boundary
condition that reusable birth-region learning should help most when regions
actually recur often enough to supply evidence.
