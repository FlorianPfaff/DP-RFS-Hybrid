# DP-specific feasibility audit

Study date: 2026-10-03. Historical implementation: `0aace33`.

The bounded study has an explicit model/novelty gate before development or
held-out tracking experiments. The current formulation fails that gate:
posterior overlap omits the prior used to obtain a track posterior, and the
tracker supplies the state at confirmation rather than at birth. The decision
applies to the current DP-specific interpretation, not to every possible DP
birth learner. Comparative performance against the proposed new baselines is
**not evaluated**. Seeds 200--299 remain unused by this study.

The mathematical audit, literature matrix, accepted conditional protocol, and
decision memo are in the [paper repository](https://github.com/FlorianPfaff/2026-07-DP-RFS-Hybrid-Paper)
under `notes/dp-feasibility-20261003/`. No filter behavior or historical
benchmark output is changed here.

## Diagnostic scope

`birth_model_audit.py` provides deterministic counterexamples and positive
controls using the original production filter code:

1. Two prior/measurement pairs produce the same posterior but different
   likelihood ratios between birth regions. The learner cannot recover this
   distinction from its current input.
2. A spatially constant likelihood does not change assignment odds; overlap
   with a prior-conditioned posterior can change them. This is an isolated
   evidence-interface test, not a claimed empty-track confirmation event.
3. Interpreting the input instead as an independent noisy observation makes
   the overlap valid, but does not justify directly pooling its measurement
   covariance into a population covariance in the low-information limit.
4. Positive controls verify the independent-observation overlap identity,
   the diffuse-prior limit, and the Gaussian mixture moment identity. The
   moment merge is legitimate for appropriately conditioned posterior
   sufficient statistics; adding posterior covariance is not itself a bug.
5. A two-scan constant-velocity track learns its confirmation position, not
   its initial position.

These are characterization tests for the audited historical formulation.
Passing them confirms reproduction of the limitations, not model validity.
When a separately authorized redesign changes those semantics, update the
diagnostics and decision together; do not preserve the limitations as required
behavior.

## Reproduction

Run on `gpuserver6000` (or `gpuserver4090`) through the configured jumpserver:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 experiments/audit_birth_model.py --source-commit COMMIT --output results/audit.json
python3 experiments/plot_birth_model_audit.py --input results/audit.json --output-prefix results/audit
```

Use the exact exported source commit, not the historical implementation hash
when reproducing the newly added diagnostic program. Source checksums and
Python/NumPy versions are recorded in the JSON. An exported archive has no
`.git` directory, so its provenance honestly records that the commit was not
verified on the server; compare the source hashes with the originating Git
checkout. Output files are not overwritten.

No new experiment-driven paper claim follows from these diagnostic checks.
The full six-condition comparison, shared learner adapter, smoothing, and
matched-delay ablations were deliberately not implemented after the early stop.
