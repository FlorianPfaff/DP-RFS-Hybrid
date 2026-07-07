# Structured Clutter Metrics Experiment

The structured clutter demo now has a paper-facing metrics script:

```bash
python experiments/structured_clutter_metrics.py --seeds 20 --scans 20 --output results/structured_clutter_metrics.csv
```

The script compares three tracker configurations:

- `fixed_scalar_clutter`: scalar clutter intensity from the birth model;
- `fixed_gmm_clutter`: hand-specified Gaussian-mixture clutter density at the known hotspot;
- `adaptive_dp_clutter`: online `DirichletProcessClutterModel` supplying posterior-predictive clutter intensities.

The CSV contains one row per random seed and tracker configuration with:

- total births;
- final estimated tracks;
- final active tracks;
- hotspot-track-step proxy count;
- final active birth atoms;
- final active clutter atoms.

The hotspot-track-step proxy counts how often confirmed estimates remain near the persistent structured-clutter hotspot. Lower is better for clutter robustness.

Generate PDF plots from the CSV with:

```bash
python experiments/plot_structured_clutter_metrics.py --input results/structured_clutter_metrics.csv --output-dir results/figures
```

The first paper-facing figure should probably use `structured_clutter_hotspot_track_steps.pdf`, with `total_births` as the second-most useful diagnostic.
