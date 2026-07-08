# Delayed DP Birth Learning

The tracker can now spawn tentative Bernoulli birth tracks without immediately updating the DP birth atom set.

## Motivation

Immediate DP birth updates are brittle: one unexplained clutter measurement can become a learned birth hotspot. Delayed learning uses the RFS-style existence state as a filter on what the DP layer is allowed to learn.

## Configuration

```python
tracker = LabeledMultiBernoulliTracker(
    ...,
    delayed_birth_learning=True,
    birth_confirmation_age=2,
    birth_confirmation_existence=0.7,
    birth_learning_weight=1.0,
)
```

## Behavior

When `delayed_birth_learning=False`, the existing immediate behavior remains:

```text
unassigned measurement -> DP birth decision -> DP atom update -> Bernoulli birth track
```

When `delayed_birth_learning=True`, the tracker uses:

```text
unassigned measurement -> DP birth preview -> Bernoulli birth track
surviving confirmed track -> DP birth atom update
```

A pending birth track updates the DP birth model only when both conditions are true:

```text
track.age >= birth_confirmation_age
track.existence >= birth_confirmation_existence
```

This is a first implementation of two-time-scale learning: the RFS layer can create and update tentative tracks every scan, while the DP birth distribution learns only from posterior-confirmed evidence.

## Current approximation

Confirmed track states are converted into weighted Gaussian birth-atom evidence using `DirichletProcessBirthModel.learn_from_state`. This is a moment-style approximation, not a full posterior over historical birth origins.

## Next improvements

- store the original birth-origin state separately from the current track state;
- use smoothed trajectory states instead of the current filtered state;
- learn with fractional newborn responsibilities from a PMBM/GLMB backend;
- separate birth rate from normalized birth density.
