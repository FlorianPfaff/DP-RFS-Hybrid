# Recurring-Birth Benchmark

## Purpose

The benchmark tests the paper's delayed-learning claim directly: recurring
targets emerge from two initially unknown spatial regions, while missed
detections and uniform Poisson clutter create misleading one-scan birth
evidence. Every method receives identical truth and measurements for each seed.

The native comparison holds the compact LMB-style tracker fixed and varies only
the birth mechanism:

- `fixed_broad_birth`: non-learning diffuse Gaussian prior;
- `measurement_driven_birth`: the measurement-driven birth rule of Lin, Vo,
  and Nordholm, approximating its unassigned probability by the greedy
  backend's unassigned measurement set;
- `dp_immediate`: updates DP atoms from every accepted birth candidate;
- `dp_delayed_append`: learns only after confirmation, but appends every state;
- `dp_delayed_recluster`: learns only after confirmation and reclusters states;
- `oracle_birth`: fixed mixture at the two true birth regions.

The official PMBM comparison changes the backend as well as the birth prior:

- `pmbm_broad`: PMBM with the same diffuse Gaussian birth prior;
- `pmbm_oracle`: PMBM with the true two-component birth mixture.

This separation matters. Native ablations isolate the proposed learning rule;
PMBM rows answer whether the compact prototype is competitive with a modern
multi-hypothesis RFS filter, but they do not isolate a single mechanism.

## Metrics

The primary metric is RMS GOSPA with `p=2`, cutoff `c=10 m`, and `alpha=2`.
Localization, missed-target, and false-target components are reported
separately. Pairwise claims use paired seeds, a 20,000-resample bootstrap
confidence interval for the mean GOSPA difference, and a 50,000-sample
sign-flip test.

The first 20 paired seeds form a development set. They select `alpha=10` from
`alpha in {1, 2, 5, 10, 20}`. Headline comparisons use the held-out 80 seeds
20--99. The full sensitivity sweep is retained alongside the primary results;
the residual branch is underweighted at the two smallest values, while
performance is stable around the selected value.

## Native Run

```bash
python -m pip install -e ".[test,benchmark]"
python experiments/recurring_birth_metrics.py \
  --seeds 100 --scans 96 \
  --output results/recurring_birth_metrics_seed100.csv
```

## Official PMBM Run

The reference run uses:

- `Agarciafernandez/MTT` commit `cc5c6df0e9343dc1345d91ed6ab95f01db4d1821`;
- `USNavalResearchLaboratory/TrackerComponentLibrary` commit
  `593ce51d64cfb79fd4293968075e5de558c83338`;
- GNU Octave 9.2.0 plus the Octave statistics package.

On a machine with Apptainer, clone those repositories to `~/MTT-reference` and
`~/TCL-reference`, then run:

```bash
experiments/run_official_pmbm_campaign.sh 100 10
```

The second argument is the number of paired seeds per parallel Octave worker.
The runner exports identical Python-generated scenarios to MAT files, runs the
official Gaussian PMBM update/prediction/pruning code, and merges all rows into
`results/recurring_birth_pmbm_seed100.csv`.

## Analysis and Figures

```bash
python experiments/analyze_recurring_birth_results.py \
  --input results/recurring_birth_metrics_seed100.csv \
          results/recurring_birth_pmbm_seed100.csv \
  --seed-start 20

python experiments/plot_recurring_birth_metrics.py \
  --input results/recurring_birth_metrics_seed100.csv \
          results/recurring_birth_pmbm_seed100.csv \
  --seed-start 20

python experiments/recurring_birth_recurrence_ablation.py
python experiments/plot_recurring_birth_recurrence.py \
  --input results/recurring_birth_recurrence_ablation_seed80.csv
```

## References

- S. Lin, B.-T. Vo, and S. E. Nordholm, "Measurement Driven Birth Model for
  the Generalized Labeled Multi-Bernoulli Filter," ICCAIS 2016,
  [arXiv:2604.03918](https://arxiv.org/abs/2604.03918).
- A. F. Garcia-Fernandez, J. L. Williams, K. Granstrom, and L. Svensson,
  "Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and
  Implementation," IEEE TAES 2018,
  [arXiv:1703.04264](https://arxiv.org/abs/1703.04264).
- A. F. Garcia-Fernandez, Y. Xia, and L. Svensson, "A Comparison Between
  PMBM Bayesian Track Initiation and Labelled RFS Adaptive Birth," FUSION
  2022, [arXiv:2207.06156](https://arxiv.org/abs/2207.06156).
- A. S. Rahmathullah, A. F. Garcia-Fernandez, and L. Svensson, "Generalized
  Optimal Sub-Pattern Assignment Metric," FUSION 2017,
  [arXiv:1601.05585](https://arxiv.org/abs/1601.05585).
