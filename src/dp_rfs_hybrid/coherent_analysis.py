"""Frozen paired-seed analysis; failures never become silently missing pairs."""
import numpy as np


def paired_summary(dp, baseline, indices=None):
    dp, baseline = np.asarray(dp, float), np.asarray(baseline, float)
    if dp.shape != baseline.shape or dp.ndim != 2 or not np.isfinite([dp, baseline]).all():
        raise ValueError("Complete finite paired seed-by-condition arrays required")
    if indices is None:
        indices = np.random.default_rng(78421).integers(len(dp), size=(10000, len(dp)))
    dp_seed, baseline_seed = dp.mean(axis=1), baseline.mean(axis=1)
    absolute = (baseline_seed-dp_seed)[indices].mean(axis=1)
    relative = 100*absolute/baseline_seed[indices].mean(axis=1)
    return {"dp": float(dp.mean()), "baseline": float(baseline.mean()),
            "improvement": float((baseline-dp).mean()),
            "improvement_percent": float(100*(baseline.mean()-dp.mean())/baseline.mean()),
            "difference_ci": np.quantile(absolute, [.025, .975]).tolist(),
            "percent_ci": np.quantile(relative, [.025, .975]).tolist()}


def verdict(primary, replay, other, correctness=True, novelty=True, failures=0):
    if failures or not correctness or not novelty:
        return "inconclusive" if failures else "no-go"
    passed = (primary["improvement_percent"] >= 3 and primary["difference_ci"][0] > 0
              and replay["difference_ci"][0] > 0
              and all(row["improvement_percent"] >= -5 for row in other))
    if passed:
        return "continue"
    ruled_out = (primary["percent_ci"][1] < 3 or replay["difference_ci"][1] <= 0
                 or any(row["improvement_percent"] < -5 for row in other))
    return "no-go" if ruled_out else "inconclusive"
