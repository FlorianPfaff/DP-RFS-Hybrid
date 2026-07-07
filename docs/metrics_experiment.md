# Structured Clutter Metrics Experiment

The structured clutter demo now has a paper-facing metrics script:

```bash
python experiments/structured_clutter_metrics.py --seeds 20 --scans 20 --output results/structured_clutter_metrics.csv
```

The script compares two tracker configurations:

- `fixed_scalar_clutter`: the baseline tracker using the scalar clutter intensity from the birth model;
- `adaptive_dp_clutter`: the same tracker with a `DirichletProcessClutterModel` supplying posterior-predictive clutter intensities.

The CSV contains one row per random seed and tracker configuration with:

- total births;
- final estimated tracks;
- final active tracks;
- hotspot-track-step proxy count;
- final active birth atoms;
- final active clutter atoms.

The hotspot-track-step proxy counts how often confirmed estimates remain near the persistent structured-clutter hotspot. Lower is better for clutter robustness.

The next paper step is to generate a plot from this CSV and commit the figure to the paper repository.
