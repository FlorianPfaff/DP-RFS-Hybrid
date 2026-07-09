# Responsibility Learning Controls

`LabeledMultiBernoulliTracker` exposes first-pass controls for slowing down nuisance-distribution learning on both the clutter and birth sides.

## Clutter responsibility gates

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

This is a first step toward two-time-scale learning: the RFS tracker can update every scan, while the DP clutter model can update more conservatively.

## Delayed DP birth learning

```python
delayed_birth_learning: bool = False
birth_confirmation_age: int = 2
birth_confirmation_existence: float = 0.7
```

In default mode, an accepted birth measurement immediately updates the DP birth model. With `delayed_birth_learning=True`, the measurement still spawns a tentative Bernoulli track, but the DP birth atoms are not updated immediately.

A tentative birth updates the DP birth model only when the track satisfies both conditions:

```text
track.age >= birth_confirmation_age
track.existence >= birth_confirmation_existence
```

Confirmed labels are reported in `StepSummary.confirmed_births`.

## Open follow-up work

- use an exponential moving average over repeated measurements or cells;
- avoid learning from measurements assigned to high-existence tracks beyond the current fractional-responsibility gate;
- recluster confirmed birth evidence against existing birth atoms instead of always appending a new atom;
- replace greedy single-scan responsibilities with multi-hypothesis responsibilities.
