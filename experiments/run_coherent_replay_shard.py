"""Schedule disjoint development blocks using the unchanged trial function."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import time

from dp_rfs_hybrid.coherent_study import CONDITIONS, configurations, configuration_id
from run_coherent_study import job, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--stop-seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.first_seed < args.stop_seed <= 50:
        raise ValueError("Only development replay seeds are allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"first_seed": args.first_seed, "stop_seed": args.stop_seed, "host": platform.node(),
                "source_hash": source_hash(), "start_unix": time.time(), "workers": args.workers,
                "inference_commit": "b688b3caf73b696692b35f73bca0ff742e7e2c81"}
    tasks = []
    for seed in range(args.first_seed, args.stop_seed):
        for condition in CONDITIONS:
            for config in configurations():
                name = f"{seed:03d}_{condition}_{configuration_id(config)}_replay_p128_r0.json"
                path = args.output/"replay"/name
                if not path.exists():
                    tasks.append((seed, condition, config, "replay", 128, 0, str(path)))
    manifest["pending"] = len(tasks)
    path = args.output/f"replay-shard-{args.first_seed}-{args.stop_seed}.json"
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            status, _ = future.result()
            failures += status != "ok"
            if status != "ok" or done % 500 == 0 or done == len(tasks):
                print(f"replay shard {args.first_seed}:{args.stop_seed}: {done}/{len(tasks)}, {failures} failures", flush=True)
    manifest.update({"end_unix": time.time(), "failures": failures})
    path.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
