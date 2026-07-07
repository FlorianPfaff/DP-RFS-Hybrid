# Responsibility Learning Controls

`LabeledMultiBernoulliTracker` now exposes two first-pass controls for slowing down DP clutter learning:

```python
clutter_responsibility_learning_rate: float = 1.0
min_clutter_responsibility_to_learn: float = 0.0
```

The raw clutter responsibility is computed from target-vs-clutter or birth-vs-clutter odds. Before updating `DirichletProcessClutterModel`, the tracker transforms it as:

```text
if responsibility < min_clutter_responsibility_to_learn:
    learned_responsibility = 0
else:
    learned_responsibility = clutter_responsibility_learning_rate * responsibility
```

This is not yet delayed confirmation or full smoothing. It is a small, explicit first step toward two-time-scale learning: the RFS tracker can update every scan, while the DP clutter model can update more conservatively.

Open follow-up work:

- use an exponential moving average over repeated measurements or cells;
- avoid learning from measurements assigned to high-existence tracks;
- delay DP birth updates until a newborn track survives several scans;
- replace greedy single-scan responsibilities with multi-hypothesis responsibilities.
