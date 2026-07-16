# Recurring-birth paired comparisons

Negative paired differences favor `dp_delayed_recluster`.

| Comparator | n | DP | Comparator | Difference (bootstrap 95% CI) | Improvement | p |
|---|---:|---:|---:|---:|---:|---:|
| fixed_broad_birth | 100 | 4.493 | 5.009 | -0.516 [-0.578, -0.453] | 10.3% | 2e-05 |
| measurement_driven_birth | 100 | 4.493 | 4.841 | -0.348 [-0.417, -0.276] | 7.2% | 2e-05 |
| dp_immediate | 100 | 4.493 | 6.679 | -2.186 [-2.391, -1.977] | 32.7% | 2e-05 |
| dp_delayed_append | 100 | 4.493 | 4.639 | -0.146 [-0.185, -0.110] | 3.2% | 2e-05 |
| oracle_birth | 100 | 4.493 | 3.854 | 0.639 [0.579, 0.702] | -16.6% | 2e-05 |
| pmbm_broad | 100 | 4.493 | 4.786 | -0.294 [-0.356, -0.227] | 6.1% | 2e-05 |
| pmbm_oracle | 100 | 4.493 | 3.296 | 1.197 [1.128, 1.268] | -36.3% | 2e-05 |
