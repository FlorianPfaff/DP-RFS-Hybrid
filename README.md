# DP-RFS Hybrid

Reference sandbox for a Dirichlet-process birth layer inside an RFS-style
multitarget tracker.

The first prototype keeps the RFS side intentionally modest: an independent
labeled multi-Bernoulli-style tracker with greedy association. The Bayesian
nonparametric part is restricted to adaptive birth modeling. Unassigned
measurements compete between clutter, reuse of an existing DP birth atom, and
creation of a new birth atom.

This repository is the implementation sandbox for the companion paper notes in
`FlorianPfaff/2026-07-DP-RFS-Hybrid-Paper`. Mature pieces can be upstreamed to
PyRecEst later in small, reviewable increments.

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

The demo simulates two recurring birth regions, clutter, and a compact tracker
that learns birth atoms online.

## Package Layout

```text
src/dp_rfs_hybrid/
  gaussian.py      # linear-Gaussian prediction, update, likelihood
  dp_birth.py      # truncated DP birth atom model
  lmb_tracker.py   # small RFS-style multi-Bernoulli tracker
examples/
  run_synthetic_birth_demo.py
tests/
```

## Current Limitations

- Greedy nearest-neighbor association, not GLMB/PMBM hypothesis management.
- Finite active DP atoms, not full posterior sampling of the random measure.
- Gaussian birth atoms and linear-Gaussian tracking only.
- Heuristic birth-vs-clutter odds threshold.

Those restrictions are intentional for the first artifact: the goal is to
isolate whether adaptive nonparametric birth structure helps before adding a
larger RFS backend.
