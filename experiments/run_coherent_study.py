"""Restartable study stages, with explicit freeze required for held-out seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np

from dp_rfs_hybrid.coherent_study import CONDITIONS, configuration_id, configurations, run_trial


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("src", "experiments", "tests"):
        for path in sorted((root/folder).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def job(arguments):
    seed, condition, config, mode, particles, algorithm_seed, destination = arguments
    try:
        result = run_trial(seed, condition, config, mode, particles, algorithm_seed)
    except Exception:
        result = {"seed": seed, "condition": condition, "config": config, "mode": mode,
                  "particles": particles, "algorithm_seed": algorithm_seed,
                  "status": "failed", "traceback": traceback.format_exc()}
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n")
    temporary.replace(path)
    return result["status"], str(path)


def select(rows, field, top=1):
    ranking = {}
    for config in configurations():
        matched = [row for row in rows if row["config"] == config]
        if not matched:
            continue
        score = float("inf") if any(r["status"] != "ok" for r in matched) else float(np.mean([r[field] for r in matched]))
        ranking.setdefault(config["learner"], []).append((score, configuration_id(config), config))
    return {name: [entry[2] for entry in sorted(entries)[:top]] for name, entries in ranking.items()}


def load_rows(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "replay", "closed", "heldout", "sensitivity"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == "pilot":
        configs = [{"learner": "dp", "alpha": 1}, {"learner": "finite", "alpha": 1, "components": 4},
                   {"learner": "kde", "bandwidth": 1., "residual": .2}]
        seeds, conditions, modes = range(1), CONDITIONS, ["replay", "closed"]
    elif stage == "replay":
        configs, seeds, conditions, modes = configurations(), range(50), CONDITIONS, ["replay"]
    elif stage == "closed":
        shortlist = json.loads((root/"shortlist.json").read_text())
        configs = [c for cs in shortlist.values() for c in cs]
        seeds, conditions, modes = range(50, 100), CONDITIONS, ["closed"]
    else:
        freeze = json.loads((root/"freeze.json").read_text())
        if freeze["source_hash"] != source_hash():
            raise RuntimeError("Frozen source hash differs; held-out execution prohibited")
        configs = list(freeze["selected"].values())
        if stage == "heldout":
            dp = freeze["selected"]["dp"]
            configs += [dict(dp, ablation=name) for name in ("prior_feedback", "confirmation_time")]
            seeds, conditions, modes = range(200, 300), CONDITIONS, ["replay", "closed"]
        else:
            configs = [c for c in configs if c["learner"] != "kde"]
            seeds, conditions, modes = range(10), CONDITIONS[:3], ["replay", "closed"]
    tasks = []
    for seed in seeds:
        for condition in conditions:
            for config in configs:
                for mode in modes:
                    for particles in ([64, 128, 256] if stage == "sensitivity" else [128]):
                        for repetition in (range(3) if stage == "sensitivity" else [0]):
                            name = f"{seed:03d}_{condition}_{configuration_id(config)}_{mode}_p{particles}_r{repetition}.json"
                            destination = root/stage/name
                            if not destination.exists():
                                tasks.append((seed, condition, config, mode, particles, repetition, str(destination)))
    manifest = {"stage": stage, "source_hash": source_hash(), "seed_range": [seeds.start, seeds.stop],
                "configurations": configs, "conditions": list(conditions), "modes": modes,
                "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
                "workers": args.workers, "pid": os.getpid(), "start_unix": time.time(), "pending": len(tasks)}
    (root/f"{stage}-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            status, path = future.result()
            failures += status != "ok"
            if status != "ok" or done % 25 == 0 or done == len(tasks):
                print(f"{stage}: {done}/{len(tasks)} completed, {failures} failures; {path}", flush=True)
    rows = load_rows(root/stage)
    if stage == "replay":
        if len(rows) != 18900:
            raise RuntimeError("Incomplete replay campaign")
        (root/"shortlist.json").write_text(json.dumps(select(rows, "log_loss", 2), indent=2)+"\n")
    elif stage == "closed":
        if len(rows) != 1800:
            raise RuntimeError("Incomplete closed-loop campaign")
        selected = {name: values[0] for name, values in select(rows, "gospa").items()}
        scores = {name: np.mean([r["gospa"] for r in rows if r["config"] == config and r["status"] == "ok"])
                  for name, config in selected.items()}
        comparator = min(("finite", "kde"), key=lambda name: scores[name])
        proposal = {"selected": selected, "comparator": comparator, "development_gospa": scores,
                    "source_hash": source_hash(), "created_unix": time.time(),
                    "heldout_seeds": [200, 299], "reserved_seeds": [300, 399]}
        (root/"freeze-proposal.json").write_text(json.dumps(proposal, indent=2)+"\n")
    print(json.dumps({"stage": stage, "total": len(rows), "failures": sum(r["status"] != "ok" for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
