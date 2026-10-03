# No-go: coherent DP birth learning

Development-selected simple comparator: **kde**.

Primary mean RMS GOSPA: DP 5.0006; comparator 4.7626.
Relative improvement -5.00% (95% paired block-bootstrap interval -6.24% to -3.78%).
Replay log-loss improvement 0.7313 nats (95% interval 0.6844 to 0.7784).

| Condition | DP RMS GOSPA | Comparator | Improvement |
| --- | ---: | ---: | ---: |
| stationary2 | 4.4330 | 4.6676 | 5.03% |
| stationary4 | 4.7615 | 4.6918 | -1.49% |
| stationary8 | 5.8072 | 4.9285 | -17.83% |
| nonrecurring | 7.5484 | 4.8095 | -56.95% |
| high_clutter | 6.5009 | 7.3685 | 11.77% |
| drift | 4.7412 | 4.6782 | -1.35% |

Correctness: analytic raw-history factors and collapsed Gaussian updates pass exact-reference checks; this is conditional inference, not a joint RFS/DP posterior.

DP-specific value is tested against the selected finite/KDE baseline, not inferred from improvements over erroneous evidence ablations.

Limitations: known fixed birth spread; hard association and admission; selection and fragmentation; approximate first-detection proposal; stationary model under drift; synthetic data and compact tracker only.

Full secondary metrics, algorithm-randomness sensitivity, and the worst paired seed in each condition are in summary.json. All 6,000 held-out rows are retained; no failed run was discarded.

No further tuning from these held-out results. Full PMBM integration remains a separately scoped next step only after a positive decision.
