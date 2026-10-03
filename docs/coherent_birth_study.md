# Coherent birth-learning viability study

The replacement is separate from `dp_birth.py`, the historical immediate and
delayed learners, and all historical benchmarks. The old audit continues to
describe those methods accurately; it does not characterize this replacement.

## Modules

- `coherent_evidence.py`: immutable raw histories; correlated linear-Gaussian
  likelihood; all feasible physical birth times; canonical information factors.
- `coherent_mixture.py`: analytically integrated Gaussian centers; collapsed
  SMC with joint allocation/time proposals and Gibbs rejuvenation; finite
  Dirichlet control; exhaustive partition/time reference for at most five histories.
- `coherent_study.py`: common density adapter, uncertainty-aware reference KDE,
  history admission, controlled scenarios, common replay and closed-loop trials.
- `coherent_analysis.py`: complete paired seed-block bootstrap and decision rule.

No region-covariance learning, forgetting, or mixture pruning is used. The
compact tracker still uses selected associations and admission decisions. Its
initialization proposal is not a full undetected-target PPP; see the derivation
in the paper repository for this and other approximation boundaries.

## Reproduction

Run calculations on `gpuserver6000` through its configured jump host, not the
desktop. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and install this checkout
or set `PYTHONPATH` to its `src` directory. Python 3.10+ and NumPy are sufficient
for inference/tests; the final figure generator uses Matplotlib.

```bash
python -m pytest tests -q --junitxml=tests.xml
python experiments/validate_coherent_birth.py --output validation.json
python experiments/run_coherent_study.py replay --output results/coherent --workers 32
python experiments/run_coherent_study.py closed --output results/coherent --workers 32
python experiments/freeze_coherent_study.py --results results/coherent \
  --validation validation.json --implementation-commit FULL_SHA \
  --paper-commit FULL_SHA --ci-url CI_RUN_URL
```

Commit `freeze.json`, the protocol, configuration selection and provenance before
running the following commands. Do not change the frozen source after seeing
held-out outcomes. The runner checks the implementation/analysis source hash.

```bash
python experiments/run_coherent_study.py heldout --output results/coherent --workers 32
python experiments/run_coherent_study.py sensitivity --output results/coherent --workers 24
python experiments/analyze_coherent_study.py --results results/coherent \
  --deliverables results/coherent/deliverables
```

Development uses 0-49 for replay shortlisting, 50-99 for closed-loop selection;
held-out uses 200-299. Seeds 300 and above are rejected by the scenario generator
to preserve the planned later 300-399 PMBM campaign. Raw JSON records are written
atomically and resumable without deleting failed trials. Complete results are
archived as deterministic gzip JSONL files with checksums, including the common
history streams and their raw observations. Truth labels in those streams are
diagnostics only and never enter learning.

Numerical failures and incomplete stages prevent positive conclusions. The
decision requires >=3% primary GOSPA improvement with a paired 95% interval
excluding zero, better replay log loss with a paired interval excluding zero,
and <=5% degradation in nonrecurrence and high clutter. Drift is descriptive.
Correctness or improvement over a deliberately broken ablation is not, by
itself, a DP-specific contribution.
