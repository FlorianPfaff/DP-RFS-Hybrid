# Delayed Birth Learning Status

Goal:

```text
track initiation != DP birth-density learning
```

Implemented first pass:

- `LabeledMultiBernoulliTracker(delayed_birth_learning=True)` enables delayed learning.
- Accepted birth measurements still spawn tentative Bernoulli tracks immediately.
- The DP birth atoms are not updated at the moment of tentative track creation.
- Pending tracks update the DP birth model only after reaching `birth_confirmation_age` and `birth_confirmation_existence`.
- Confirmed birth evidence is reclustered against existing birth atoms; nearby
  confirmed births reuse one atom, while distant confirmed births create
  separate atoms.
- `StepSummary.confirmed_births` reports which labels triggered delayed DP birth learning.

Current limitations:

- confirmation uses the current filtered state, not a trajectory smoother;
- there is no explicit negative evidence from tentative births that die early.

This is enough to test the two-time-scale idea without changing the default immediate-learning behavior.
