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

Status: completed, **no-go for the DP-specific contribution under the frozen
protocol**. All 18,900 development replay, 1,800 closed-loop development, 6,000
held-out, and 1,080 sensitivity trials completed without numerical failures.
`freeze.json` was committed in `169cd6a` before held-out execution; it locks the
source/analysis hash, configurations, comparator, correctness artifact, and
protocol commit.

Selected DP: alpha 5. Selected finite mixture: K=16, alpha 1. Selected KDE:
positional bandwidth 1, velocity bandwidth 0.1, residual 0.5. The development
GOSPA means over all six conditions are 5.64610, 7.19086, and 5.19322 respectively;
KDE is therefore the locked simple comparator. These are development numbers,
not validation claims.

`development-replay.tar.gz` and `development-closed.tar.gz` preserve every raw
development record and the corresponding manifests before held-out evaluation.
The former also includes common raw histories and both pilot versions.

## Frozen result

Primary RMS GOSPA: DP 5.000569 versus KDE 4.762623, a 4.996% degradation
(paired 95% interval: 3.785% to 6.242% worse). Replay predictive log loss:
DP 6.259396 versus KDE 6.990696, a 0.731299-nat improvement (95% interval:
0.684441 to 0.778364). Nonrecurrence GOSPA degrades by 56.947%, independently
failing the 5% guardrail. No further tuning or PMBM study was started.

`evaluation/summary.json` contains all condition/learner metrics, uncertainty,
algorithm-randomness sensitivity, and representative failed tracking cases.
`evaluation/*.jsonl.gz` retains every raw record, including 900 common-history
streams. `evaluation/archives.json` provides row counts and SHA-256 checksums.
`evaluation/decision.md` is the frozen analysis output; the paper repository
contains the interpreted decision memo and additional uncertainty figure.

Held-out runs used gpuserver4090 (Python 3.10.12, NumPy 2.2.6); sensitivity and
analysis used gpuserver6000 (Python 3.12.3, NumPy 2.0.2). Wall-clock timings are
concurrent-run observations, not controlled cross-host speed benchmarks.
A transfer was initially extracted before its upload completed; that partial
copy was replaced from the completed archive and verified with `tar -d` before
analysis. This transport retry did not rerun or select model outcomes.
Matplotlib emitted a warning about unavailable Axes3D on the analysis host;
the generated two-dimensional figures were inspected and are unaffected.

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
