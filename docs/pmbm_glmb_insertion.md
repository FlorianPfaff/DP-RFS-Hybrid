# PMBM/GLMB Posterior-Predictive Insertion Sketch

The current implementation is intentionally LMB-like with greedy association. The next RFS backend should not expose DP component assignments to every PMBM/GLMB global hypothesis. The intended abstraction is posterior-predictive insertion.

## Core interface

An RFS backend should request only scalar intensities or predictive densities:

```python
birth_intensity(x) -> float
clutter_intensity(z) -> float
target_likelihood(z, x, label=None) -> float
```

The backend should not need to know which DP atom produced a birth or clutter measurement. DP atom assignments remain inside the Bayesian-nonparametric layer.

## Birth insertion

For a PMBM-style backend, the birth PPP intensity should be supplied as

```text
D_B(x) = lambda_B * E_q[ integral K_B(x | theta) dG_B(theta) ]
```

where the DP layer models the normalized birth density and a separate rate model supplies `lambda_B`.

## Clutter insertion

For a PMBM/GLMB update, each measurement needs a clutter intensity value:

```text
kappa(z) = lambda_C * E_q[ integral K_C(z | psi) dG_C(psi) ]
```

The association machinery receives `kappa(z)` as a scalar. It should not branch over clutter atoms.

## Feedback to the DP layer

After the RFS update, the backend should expose soft responsibilities:

```text
r_C(z)      = Pr(z is clutter | Z_1:k)
r_B(z)      = Pr(z initiates a new target | Z_1:k)
r_label(z)  = Pr(z belongs to label | Z_1:k)
```

The DP layer then performs weighted updates:

```text
DP clutter update: use z with weight r_C(z)
DP birth update: use confirmed newborn state with weight/smoothing from r_B
HDP motion update: use mode counts from surviving track posteriors
```

## What to avoid

Do not construct hypotheses of the form:

```text
association hypothesis x DP atom assignment hypothesis
```

That multiplication would destroy most of the practical pruning structure of GLMB/PMBM trackers. The DP component assignment should be marginalized inside the predictive intensity before the RFS update.

## Current prototype correspondence

The LMB-style prototype already follows a small version of this idea:

- `DirichletProcessClutterModel.intensity(z)` supplies a scalar local clutter intensity;
- `DirichletProcessBirthModel.process(..., clutter_intensity=...)` uses that local intensity in birth-vs-clutter decisions;
- the tracker feeds fractional clutter responsibilities back into the clutter model.

The PMBM/GLMB extension should preserve this direction while replacing greedy association with a proper hypothesis manager.
