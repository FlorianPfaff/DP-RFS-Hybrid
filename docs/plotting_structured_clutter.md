# Plotting Structured Clutter Metrics

Generate the CSV first:

```bash
python experiments/structured_clutter_metrics.py --seeds 20 --scans 20 --output results/structured_clutter_metrics.csv
```

Then generate PDF figures:

```bash
python experiments/plot_structured_clutter_metrics.py --input results/structured_clutter_metrics.csv --output-dir results/figures
```

The plotting script expects `matplotlib` in the active Python environment.

The metrics script currently compares three tracker configurations:

- `fixed_scalar_clutter`: scalar clutter intensity baseline;
- `fixed_gmm_clutter`: hand-specified Gaussian-mixture clutter baseline;
- `adaptive_dp_clutter`: online DP clutter density with fractional responsibility updates.

The most paper-relevant first plot is:

```text
structured_clutter_hotspot_track_steps.pdf
```

It summarizes the false-track pressure near the persistent clutter hotspot.
