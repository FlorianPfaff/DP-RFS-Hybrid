# Delayed DP Birth Learning

The first implementation updated the DP birth model immediately when an unassigned measurement passed the birth-vs-clutter odds test. That is simple, but it can learn one-scan clutter as a birth hotspot.

The next feature should separate two operations:

```text
track initiation  !=  DP birth-density learning
```

A measurement can still spawn a tentative Bernoulli track immediately, but the DP birth atoms should only be updated after the newborn track survives for several scans and reaches a sufficiently high existence probability.

## Intended API sketch

```python
tracker = LabeledMultiBernoulliTracker(
    ...,
    delayed_birth_learning=True,
    birth_confirmation_age=2,
    birth_confirmation_existence=0.7,
)
```

When delayed learning is enabled:

1. score an unassigned measurement with `birth_model.decide(...)`;
2. if accepted, spawn a tentative track from the predicted birth state;
3. store pending birth evidence on the track;
4. do not update `birth_model.atoms` immediately;
5. after the track is old enough and has high posterior existence, update the DP birth model with the track's smoothed state.

## Why this matters

This is the birth-side analogue of clutter responsibility gating. The RFS layer should first establish that a physical target likely exists. The DP layer should then learn birth structure from that posterior evidence, not from raw one-scan detections.

## Open design choices

- Whether confirmed evidence updates the originally selected atom or reclusters against the current birth atoms.
- Whether confirmation should use the original birth measurement, the current filtered state, or a short trajectory smoother.
- Whether false tentative births should provide negative evidence to the birth model or only fail to update it.
