# Locked recurring-birth evaluation provenance

The method, scenario, parameters, and reporting protocol were frozen after the
seeds 0--99 development campaign. Fresh paired seeds 100--199 were then run for
the paper's headline evaluation. The locked campaign was started on 2026-07-16
without inspecting any result in that seed range beforehand.

- Implementation freeze: commit `0bd0f9d` contains the delayed confirmed-state
  reclustering implementation and benchmark definitions. Subsequent campaign
  changes add seed-range orchestration, grouped analysis, documentation, and
  result artifacts without changing tracker behavior.
- Scenario: 96 scans, two alternating recurring birth regions, 12 targets per
  realization, 14-scan target lifetimes, detection probability 0.9, and uniform
  Poisson clutter with mean 6 measurements per scan.
- Evaluation seeds: 100--199 for every native, PMBM, and recurrence cell.
- Metric: sequence RMS GOSPA with order 2, cutoff 10 m, and alpha 2.
- Mean uncertainty: 1.96 standard errors over 100 realizations.
- Pairwise uncertainty: 20,000 paired bootstrap resamples of the mean
  sequence-level RMS GOSPA difference.
- Pairwise test: 50,000 random paired sign flips; the smallest reportable Monte
  Carlo p-value is approximately `2e-5`.
- Native computation: `gpuserver6000`, using its system Python scientific
  environment. All six variants use the same compact greedy LMB-style backend.
- PMBM computation: `gpuserver4090`, using GNU Octave 9.2.0 in the
  `gnuoctave/octave:9.2.0` Apptainer image.
- PMBM reference: `Agarciafernandez/MTT` commit
  `cc5c6df0e9343dc1345d91ed6ab95f01db4d1821`.
- Assignment dependency: `USNavalResearchLaboratory/TrackerComponentLibrary`
  commit `593ce51d64cfb79fd4293968075e5de558c83338`.

The native rows isolate the birth mechanism. The PMBM rows change the outer
filter as well and are an algorithm-level reference. The oracle rows use the
true two-component birth mixture and are diagnostic rather than deployable.

The recurrence ablation varies the spawn interval over 6, 8, 12, and 16 scans,
corresponding to 16, 12, 8, and 6 target birth events. It holds all tracker
parameters fixed and uses the same locked seeds 100--199.

## Artifact checksums

```text
66cf44428c2b94c27913f50ef3ade6e2c95762e017df546a5957d8d4bbe5f4f0  recurring_birth_metrics_seed100_199.csv
74c1e9b65258874adfde6f9664340fd960779c08af572a7efa82af6b010f311d  recurring_birth_pmbm_seed100_199.csv
0d09ee45f68c273f48723687c0bf973e7a9c81cb28ceddbb3608fa3a50b62428  recurring_birth_recurrence_ablation_seed100_199.csv
ea6baddcb495f96ba977f69a1de66df3ba6c4baf262b081e016f1478bfdb0438  recurring_birth_locked_summary.csv
b8fc1915d675dce9661b668b9c831c1432de487593681297b54296afa6b2fe94  recurring_birth_locked_paired_comparisons.csv
ed912a0c14d3b876c4bd5df4657cedfec8b560d98b80b443ae0d14b15adbe30b  recurring_birth_locked_recurrence_comparisons.csv
```
