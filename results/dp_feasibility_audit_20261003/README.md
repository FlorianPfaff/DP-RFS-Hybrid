# Stage-one model audit results

Study date: 2026-10-03. Decision: **no-go for the current DP-specific
formulation**. This is a mathematical/model assessment supported by
deterministic diagnostics, not an empirical comparison with the unimplemented
mixture and KDE baselines.

## Execution

- Source export: `d51aa81f95505b7605953cbe0ec17313c2755b00`.
- Historical algorithm audited: `0aace330b2bfb3c8a9c3b23eed4a54090364e955`.
- SSH alias: `gpuserver6000`, reached through the configured jumpserver.
- Server-reported hostname: `workstation2`.
- Run directory: `/home/florianpfaff/dp-rfs-feasibility-20261003`.
- Python 3.12.3; NumPy 2.0.2.
- Full test suite: **38 passed**, including five new characterization tests.
- [Hosted CI](https://github.com/FlorianPfaff/DP-RFS-Hybrid/actions/runs/37133969927)
  passed for Python 3.10, 3.11, and 3.12 at the source export commit.
- Diagnostic and plotting processes exited successfully. Matplotlib warned
  that Axes3D was unavailable; the generated figure is entirely two-dimensional.
- No development sweep, baseline comparison, bootstrap analysis or locked
  tracking campaign was run. Seeds 200--299 remain unused by this study.

The server ran a Git archive without Git metadata. Consequently the raw JSON
correctly reports `commit_verified_locally: false`. All eleven source checksums
recorded there were independently checked against the originating committed
checkout using:

```bash
jq -r '.provenance.source_sha256 | to_entries[] | "\(.value)  \(.key)"' results/dp_feasibility_audit_20261003/diagnostics.json | sha256sum -c
```

The archive SHA-256 was
`7ff9287bdc8201a1f9ab827a0e3b96ca28a1f96fba5d6e1da9a2d8159fad0ec1`.

## Files and checksums

- `diagnostics.json`: raw counterexamples, positive controls and provenance;
  SHA-256 `d11fe630300b112b6bf793d8c368dba4a969cd89a7c7c397480c597edb53a44c`.
- `tests.xml`: JUnit record of the server test run;
  SHA-256 `19b67ce94f857add327f364c96d9bd7860a07943be68614222145a2be669b371`.
- `decision.json`: human audit disposition and explicit unexecuted stages;
  this is not an automated statistical decision.

The explanatory figure and mathematical interpretation are committed to the
[paper repository](https://github.com/FlorianPfaff/2026-07-DP-RFS-Hybrid-Paper/tree/main/notes/dp-feasibility-20261003).
No source filter behavior or historical results were changed.
