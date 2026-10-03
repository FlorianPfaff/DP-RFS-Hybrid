"""Frozen statistics, raw-record archives, figures, and short decision memo."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from dp_rfs_hybrid.coherent_analysis import paired_summary, verdict
from dp_rfs_hybrid.coherent_study import CONDITIONS, configuration_id
from run_coherent_study import load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--deliverables", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.results, args.deliverables
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((root/"freeze.json").read_text())
    rows = load_rows(root/"heldout")
    failures = [r for r in rows if r["status"] != "ok"]
    if len(rows) != 6000:
        raise RuntimeError(f"Expected 6000 held-out rows, found {len(rows)}; no partial-case conclusions")
    selected = freeze["selected"]
    records = {(r["seed"], r["condition"], configuration_id(r["config"]), r["mode"]): r for r in rows}
    if len(records) != len(rows):
        raise RuntimeError("Duplicate trial keys")

    def array(config, mode, field):
        values = []
        for seed in range(200, 300):
            block = []
            for condition in CONDITIONS:
                row = records[(seed, condition, configuration_id(config), mode)]
                if row["status"] != "ok" or row[field] is None:
                    raise RuntimeError("A confirmatory pair failed; no deletion or imputation is allowed")
                block.append(row[field])
            values.append(block)
        return np.asarray(values)

    summary = {"failures": failures, "freeze": freeze}
    if failures:
        summary["decision"] = "inconclusive"
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        (out/"decision.md").write_text("# Inconclusive\n\nNumerical failures prevent a complete paired evaluation. All failures are retained in summary.json.\n")
    else:
        dp = array(selected["dp"], "closed", "gospa")
        base = array(selected[freeze["comparator"]], "closed", "gospa")
        primary = paired_summary(dp[:, :3], base[:, :3])
        replay = paired_summary(array(selected["dp"], "replay", "log_loss")[:, :3],
                                 array(selected[freeze["comparator"]], "replay", "log_loss")[:, :3])
        comparisons = [paired_summary(dp[:, i:i+1], base[:, i:i+1]) for i in range(6)]
        summary.update({"primary": primary, "replay": replay, "conditions": dict(zip(CONDITIONS, comparisons)),
                        "decision": verdict(primary, replay, comparisons[3:5])})
        all_configs = dict(selected)
        all_configs.update({name: dict(selected["dp"], ablation=name) for name in ["prior_feedback", "confirmation_time"]})
        table = {}
        metrics = ["gospa", "log_loss", "missed", "false", "confirmation_delay", "unconfirmed_targets",
                   "fragmentation", "birth_time_error", "model_size", "runtime", "admitted", "initiations"]
        for name, config in all_configs.items():
            table[name] = {}
            for condition in CONDITIONS:
                group = [r for r in rows if r["config"] == config and r["condition"] == condition and r["mode"] == "closed"]
                values = {}
                for field in metrics:
                    available = [r[field] for r in group if r[field] is not None]
                    values[field] = {"mean": float(np.mean(available)) if available else None, "defined_runs": len(available)}
                histories = [h for r in group for h in r["histories"]]
                values["history_attribution"] = {"total": len(histories), "clutter_only": sum(h["target"] < 0 for h in histories),
                                                   "mixed": sum(h["purity"] < 1 for h in histories)}
                values["replay_log_loss"] = float(array(config, "replay", "log_loss")[:, CONDITIONS.index(condition)].mean())
                table[name][condition] = values
        summary["table"] = table
        summary["failure_examples"] = []
        for i, condition in enumerate(CONDITIONS):
            index = int(np.argmax(dp[:, i]-base[:, i]))
            seed = index+200
            summary["failure_examples"].append({"seed": seed, "condition": condition, "dp_gospa": float(dp[index, i]),
                                                "baseline_gospa": float(base[index, i]),
                                                "dp": records[(seed, condition, configuration_id(selected["dp"]), "closed")],
                                                "baseline": records[(seed, condition, configuration_id(selected[freeze["comparator"]]), "closed")]})
        sensitivity = load_rows(root/"sensitivity")
        summary["sensitivity"] = {"rows": len(sensitivity), "failures": sum(r["status"] != "ok" for r in sensitivity)}
        sensitivity_summary = []
        for name in ["dp", "finite"]:
            for mode in ["replay", "closed"]:
                field = "log_loss" if mode == "replay" else "gospa"
                for particles in [64, 128, 256]:
                    group = [r for r in sensitivity if r["config"]["learner"] == name and r["mode"] == mode and r["particles"] == particles]
                    if group and all(r["status"] == "ok" for r in group):
                        values = np.array([[next(r[field] for r in group if r["seed"] == seed and r["condition"] == condition and r["algorithm_seed"] == rep)
                                            for rep in range(3)] for seed in range(10) for condition in CONDITIONS[:3]])
                        sensitivity_summary.append({"learner": name, "mode": mode, "particles": particles,
                                                     "mean": float(values.mean()), "mean_within_case_sd": float(values.std(axis=1, ddof=1).mean())})
        summary["sensitivity"]["summary"] = sensitivity_summary
        if len(sensitivity) != 1080 or summary["sensitivity"]["failures"]:
            summary["decision"] = "inconclusive"
        (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        lines = [f"# {summary['decision'].capitalize()}: coherent DP birth learning", "",
                 f"Development-selected simple comparator: **{freeze['comparator']}**.", "",
                 f"Primary mean RMS GOSPA: DP {primary['dp']:.4f}; comparator {primary['baseline']:.4f}.",
                 f"Relative improvement {primary['improvement_percent']:.2f}% (95% paired block-bootstrap interval {primary['percent_ci'][0]:.2f}% to {primary['percent_ci'][1]:.2f}%).",
                 f"Replay log-loss improvement {replay['improvement']:.4f} nats (95% interval {replay['difference_ci'][0]:.4f} to {replay['difference_ci'][1]:.4f}).", "",
                 "| Condition | DP RMS GOSPA | Comparator | Improvement |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {c} | {s['dp']:.4f} | {s['baseline']:.4f} | {s['improvement_percent']:.2f}% |" for c,s in zip(CONDITIONS, comparisons)]
        lines += ["", "Correctness: analytic raw-history factors and collapsed Gaussian updates pass exact-reference checks; this is conditional inference, not a joint RFS/DP posterior.",
                  "", "DP-specific value is tested against the selected finite/KDE baseline, not inferred from improvements over erroneous evidence ablations.",
                  "", "Limitations: known fixed birth spread; hard association and admission; selection and fragmentation; approximate first-detection proposal; stationary model under drift; synthetic data and compact tracker only.",
                  "", "Full secondary metrics, algorithm-randomness sensitivity, and the worst paired seed in each condition are in summary.json. All 6,000 held-out rows are retained; no failed run was discarded.",
                  "", "No further tuning from these held-out results. Full PMBM integration remains a separately scoped next step only after a positive decision."]
        (out/"decision.md").write_text("\n".join(lines)+"\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        colors = {"dp": "#137C66", "finite": "#365EC9", "kde": "#C15138"}
        for name, config in selected.items():
            for ax, mode, field in [(axes[0], "closed", "gospa"), (axes[1], "replay", "log_loss")]:
                values = array(config, mode, field)
                ax.plot(range(6), values.mean(axis=0), "o-", label=name, color=colors[name])
                ax.set_xticks(range(6), ["2 regions", "4 regions", "8 regions", "New regions", "Clutter", "Drift"], rotation=25)
                ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Mean sequence RMS GOSPA")
        axes[1].set_ylabel("Pre-birth replay log loss (nats)")
        axes[0].legend(frameon=False)
        for suffix in ["png", "pdf"]:
            fig.savefig(out/f"matched-comparisons.{suffix}", dpi=180)
        plt.close(fig)
    archives = {}
    for stage in ["pilot-before-admission-fix", "pilot", "histories", "replay", "closed", "heldout", "sensitivity"]:
        paths = sorted((root/stage).glob("*.json"))
        if not paths:
            continue
        destination = out/f"{stage}.jsonl.gz"
        with destination.open("wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as zipped:
                for path in paths:
                    zipped.write(path.read_bytes().rstrip()+b"\n")
        archives[stage] = {"rows": len(paths), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (out/"archives.json").write_text(json.dumps(archives, indent=2)+"\n")
    print(json.dumps({"decision": summary["decision"], "primary": summary.get("primary"), "replay": summary.get("replay")}), flush=True)


if __name__ == "__main__":
    main()
