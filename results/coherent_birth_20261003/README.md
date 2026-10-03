# Coherent birth-learning campaign provenance

Compute host: `gpuserver6000` (actual hostname `workstation2`), through the
configured jump server. CPU only, one BLAS thread per worker.

Implementation: `b688b3caf73b696692b35f73bca0ff742e7e2c81`.
Initial implementation: `5eaa3ae5acdde2696697fdd75b8ad30aeaa715f4`.
Paper protocol revision: see `notes/coherent-birth-20261003` in the paper repo.

Validation: 53 tests pass. `validation.json` records 64 independent runs for
each exact-reference setting, including a four-dimensional history case.
The maximum mean probability error at 128 particles is 0.012842, below 0.03.

[Hosted CI](https://github.com/FlorianPfaff/DP-RFS-Hybrid/actions/runs/37143388839)
passed for Python 3.10, 3.11, and 3.12. An earlier CI failure exposed removal of
NumPy's deprecated `trapz` API; compatibility was fixed before development.

The first seed-0 pilot exposed a common admission threshold that admitted no
high-clutter tracks. Its records are preserved separately. The threshold was
repaired for every learner before the single development campaign. No old
benchmark or historical result was overwritten.

Status: development replay running; held-out data not yet opened. The frozen
configuration manifest and final records will be added incrementally.
