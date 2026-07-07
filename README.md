# DP-RFS Hybrid

Reference sandbox for Dirichlet-process layers inside RFS-style multitarget trackers.

The first prototype keeps the RFS side intentionally modest: an independent labeled multi-Bernoulli-style tracker with greedy association. The Bayesian nonparametric side now contains two deliberately separated models:

- a DP birth model for reusable birth regions inferred from unexplained measurements;
- a DP clutter density model for structured measurement-space clutter learned from soft clutter responsibilities.

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

where `DirichletProcessClutterModel` estimates `c(z)` and stores `lambda_C` as `rate`.

## Install

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Quick Demo

```bash
python examples/run_synthetic_birth_demo.py
```

The demo simulates two recurring birth regions, clutter, and a compact tracker that learns birth atoms online.

## Package Layout

```text
src/dp_rfs_hybrid/
  gaussian.py      # linear-Gaussian prediction, update, likelihood
  dp_birth.py      # truncated DP birth atom model
  dp_clutter.py    # truncated DP clutter density model
  lmb_tracker.py   # small RFS-style multi-Bernoulli tracker
examples/
  run_synthetic_birth_demo.py
tests/
```

## Current Limitations

- Greedy nearest-neighbor association, not GLMB/PMBM hypothesis management.
- Finite active DP atoms, not full posterior sampling of the random measure.
- Gaussian birth/clutter atoms and linear-Gaussian tracking only.
- Heuristic birth-vs-clutter odds threshold.
- DP clutter is currently a standalone density model; the tracker does not yet feed posterior clutter responsibilities into it automatically.

Those restrictions are intentional for the first artifact: the goal is to isolate whether adaptive nonparametric nuisance structure helps before adding a larger RFS backend.
