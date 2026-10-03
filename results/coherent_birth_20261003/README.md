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

Status: all 18,900 development replay and 1,800 closed-loop development trials
completed without numerical failures. Held-out data have not been opened at
the time of this freeze commit. `freeze.json` locks the source/analysis hash,
selected configurations, comparator, correctness artifact, and protocol commit.

Selected DP: alpha 5. Selected finite mixture: K=16, alpha 1. Selected KDE:
positional bandwidth 1, velocity bandwidth 0.1, residual 0.5. The development
GOSPA means over all six conditions are 5.64610, 7.19086, and 5.19322 respectively;
KDE is therefore the locked simple comparator. These are development numbers,
not validation claims.

`development-replay.tar.gz` and `development-closed.tar.gz` preserve every raw
development record and the corresponding manifests before held-out evaluation.
The former also includes common raw histories and both pilot versions.

## Scheduling provenance

The initial 32-worker replay pool on gpuserver6000 was deliberately stopped to
split the same campaign into disjoint seed blocks: 0-19 on gpuserver6000 and
20-49 on gpuserver4090. Completed JSON records were retained and skipped on
restart. Interrupted in-flight trials were restarted from their original RNG
seeds; no completed outcome was selected or discarded. The initial pool reports
`BrokenProcessPool` because its workers were explicitly terminated for this
scheduling change, not because a numerical method failed. The original stage
manifest is preserved as `replay-initial-manifest.json`.

The shard scheduler was added in `c6997db`; it invokes the unchanged trial
function from `b688b3c`. The two disjoint record sets are merged before any
shortlisting. Source hashes distinguish this scheduling-only addition from
the initial execution bundle.
