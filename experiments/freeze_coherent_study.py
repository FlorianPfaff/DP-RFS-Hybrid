"""Materialize the held-out lock, to be committed before evaluation starts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from run_coherent_study import source_hash, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--paper-commit", required=True)
    parser.add_argument("--ci-url", required=True)
    args = parser.parse_args()
    destination = args.results/"freeze.json"
    if destination.exists() or (args.results/"heldout").exists():
        raise RuntimeError("Refusing to replace an existing freeze or opened held-out stage")
    for commit in [args.implementation_commit, args.paper_commit]:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Full Git commit identifiers required")
    validation = json.loads(args.validation.read_text())
    if not validation["passed"]:
        raise RuntimeError("Correctness gate not passed")
    for stage, expected in [("replay", 18900), ("closed", 1800)]:
        rows = load_rows(args.results/stage)
        if len(rows) != expected or any(r["status"] != "ok" for r in rows):
            raise RuntimeError(f"Incomplete or failed development stage: {stage}")
    freeze = json.loads((args.results/"freeze-proposal.json").read_text())
    if freeze["source_hash"] != source_hash():
        raise RuntimeError("Source changed since final development stage")
    freeze.update({"implementation_commit": args.implementation_commit, "paper_commit": args.paper_commit,
                   "ci_url": args.ci_url, "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
                   "locked_unix": time.time(), "instruction": "Commit this file before running heldout; never retune from heldout."})
    destination.write_text(json.dumps(freeze, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
