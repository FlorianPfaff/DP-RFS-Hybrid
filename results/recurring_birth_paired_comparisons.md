# Recurring-birth paired comparisons

Negative paired differences favor `dp_delayed_recluster`.

| Comparator | n | DP | Comparator | Difference (bootstrap 95% CI) | Improvement | p |
|---|---:|---:|---:|---:|---:|---:|
| fixed_broad_birth | 80 | 4.458 | 4.961 | -0.502 [-0.572, -0.430] | 10.1% | 2e-05 |
| measurement_driven_birth | 80 | 4.458 | 4.843 | -0.385 [-0.450, -0.318] | 7.9% | 2e-05 |
| dp_immediate | 80 | 4.458 | 6.716 | -2.258 [-2.471, -2.039] | 33.6% | 2e-05 |
| dp_delayed_append | 80 | 4.458 | 4.584 | -0.126 [-0.182, -0.076] | 2.8% | 2e-05 |
| oracle_birth | 80 | 4.458 | 3.810 | 0.649 [0.580, 0.719] | -17.0% | 2e-05 |
| pmbm_broad | 80 | 4.458 | 4.772 | -0.314 [-0.385, -0.239] | 6.6% | 2e-05 |
| pmbm_oracle | 80 | 4.458 | 3.234 | 1.224 [1.151, 1.299] | -37.9% | 2e-05 |
