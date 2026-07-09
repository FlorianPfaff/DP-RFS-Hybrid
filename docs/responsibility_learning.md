# Responsibility Learning Controls

`LabeledMultiBernoulliTracker` exposes controls for slowing down Bayesian-nonparametric nuisance-model learning. The RFS tracker can still update every scan, while the DP birth and clutter layers learn only from more credible evidence.

## Clutter responsibility gating

For clutter, the current controls are:

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

## Delayed birth learning

For birth, the current controls are:

```python
delayed_birth_learning: bool = False
birth_confirmation_age: int = 2
birth_confirmation_existence: float = 0.7
```

When `delayed_birth_learning` is disabled, an accepted birth measurement immediately updates the DP birth model and spawns a Bernoulli track.

When `delayed_birth_learning` is enabled, an accepted birth measurement only spawns a tentative Bernoulli track. The DP birth model is not updated immediately. The track carries `pending_birth_learning=True` until it satisfies:

```text
track.age >= birth_confirmation_age
track.existence >= birth_confirmation_existence
```

Only then does the tracker add the confirmed track state as DP birth evidence. This reduces the feedback loop where one-scan clutter is learned as a recurring birth hotspot.

Open follow-up work:

- use an exponential moving average over repeated measurements or cells;
- avoid learning from measurements assigned to high-existence tracks;
- use smoothed birth-origin states instead of the current confirmed single-track state;
- replace greedy single-scan responsibilities with multi-hypothesis responsibilities.
