# DP-RFS Hybrid

Reference sandbox for Dirichlet-process layers inside RFS-style multitarget trackers.

The first prototype keeps the RFS side intentionally modest: an independent labeled multi-Bernoulli-style tracker with greedy association. The Bayesian nonparametric side now contains two deliberately separated models:

- a DP birth model for reusable birth regions inferred from unexplained measurements;
- clutter density models for structured measurement-space clutter, including a fixed Gaussian-mixture baseline and an adaptive DP clutter model learned from soft clutter responsibilities.

This repository is the implementation sandbox for the companion paper notes in `FlorianPfaff/2026-07-DP-RFS-Hybrid-Paper`. Mature pieces can be upstreamed to PyRecEst later in small, reviewable increments.

## Design principle

The RFS layer estimates physical targets: existence, labels, survival, death, missed detections, measurement-to-track association, and trajectory continuity.

The DP layer estimates unknown distributions used by the RFS layer. DP atoms are not targets. In this repository:

```text
DP birth atoms   = recurring birth regions
DP clutter atoms = recurring clutter regimes
```

For clutter, the implementation keeps the normalized density and the Poisson rate separate:

```text
kappa(z) = lambda_C * c(z)
```

where `DirichletProcessClutterModel` estimates `c(z)` and stores `lambda_C` as `rate`. `FixedGaussianMixtureClutterModel` implements the same density/intensity interface without learning, so it can be used as a hand-specified baseline.

The tracker can optionally consume a clutter model: association odds, Bernoulli existence updates, and birth decisions use `clutter_model.intensity(z)` instead of a fixed scalar clutter intensity. The tracker then feeds fractional clutter responsibilities back into adaptive clutter models. Responsibility learning can be attenuated or gated via `clutter_responsibility_learning_rate` and `min_clutter_responsibility_to_learn`.

Birth learning can also run on two time scales. With `delayed_birth_learning=True`, an accepted birth measurement still spawns a tentative Bernoulli track immediately, but the DP birth atoms are updated only after that track reaches `birth_confirmation_age` and `birth_confirmation_existence`.

## Install

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[test]"
pytest
```

The plotting script expects `matplotlib` in the active environment.

## Quick Demos

DP birth learning:

```bash
python examples/run_synthetic_birth_demo.py
```

Structured DP clutter learning:

```bash
python examples/run_structured_clutter_demo.py
```

The birth demo simulates two recurring birth regions, clutter, and a compact tracker that learns birth atoms online. The clutter demo compares fixed scalar clutter against an adaptive DP clutter model around a persistent measurement-space hotspot.

## Metrics and plots

Generate a multi-seed CSV for the structured-clutter comparison:

```bash
python experiments/structured_clutter_metrics.py --seeds 20 --scans 20 --output results/structured_clutter_metrics.csv
```

The metrics script compares:

```text
fixed_scalar_clutter
fixed_gmm_clutter
adaptive_dp_clutter
```

Generate PDF plots from the CSV:

```bash
python experiments/plot_structured_clutter_metrics.py --input results/structured_clutter_metrics.csv --output-dir results/figures
```

The most paper-relevant first plot is `structured_clutter_hotspot_track_steps.pdf`, which summarizes false-track pressure near the persistent clutter hotspot.

## Reusable Experiment API

The structured-clutter demo is backed by a reusable experiment helper:

```python
from dp_rfs_hybrid import run_structured_clutter_experiment

result = run_structured_clutter_experiment(scans=20, seed=11)
rows = result.as_rows()
```

The result reports cumulative births, final active tracks, active clutter atoms, and per-scan records that can be exported to paper tables or figures.

## Package Layout

```text
src/dp_rfs_hybrid/
  gaussian.py      # linear-Gaussian prediction, update, likelihood
  dp_birth.py      # truncated DP birth atom model
  dp_clutter.py    # fixed GMM and DP clutter density models
  lmb_tracker.py   # small RFS-style multi-Bernoulli tracker
  experiments.py   # reusable structured-clutter experiment harness
examples/
  run_synthetic_birth_demo.py
  run_structured_clutter_demo.py
experiments/
  structured_clutter_metrics.py
  plot_structured_clutter_metrics.py
tests/
```

## Current Limitations

- Greedy nearest-neighbor association, not GLMB/PMBM hypothesis management.
- Finite active DP atoms, not full posterior sampling of the random measure.
- Gaussian birth/clutter atoms and linear-Gaussian tracking only.
- Heuristic birth-vs-clutter odds threshold.
- DP clutter feedback is still approximate and single-scan; it does not yet use delayed confirmation, smoothing, or a full multi-hypothesis responsibility calculation.

Those restrictions are intentional for the first artifact: the goal is to isolate whether adaptive nonparametric nuisance structure helps before adding a larger RFS backend.
