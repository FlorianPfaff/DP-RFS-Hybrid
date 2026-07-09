# Delayed Birth Learning Status

This file tracks the implementation direction after the initial design note.

Goal:

```text
track initiation != DP birth-density learning
```

The next implementation branch should add a small first-pass mechanism to spawn tentative Bernoulli tracks from accepted birth measurements while postponing DP birth atom updates until the track has survived long enough and has sufficiently high existence probability.
