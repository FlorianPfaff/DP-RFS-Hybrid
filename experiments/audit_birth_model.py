"""Run the bounded feasibility study's deterministic model checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from dp_rfs_hybrid.birth_model_audit import run_birth_model_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audited_files = sorted((root / "src" / "dp_rfs_hybrid").glob("*.py"))
    audited_files += [Path(__file__).resolve()]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True,
    )
    if git_head.returncode == 0 and git_head.stdout.strip() != args.source_commit:
        raise ValueError("source commit does not match the checkout")
    result = run_birth_model_audit()
    result["provenance"] = {
        "source_commit": args.source_commit,
        "commit_verified_locally": git_head.returncode == 0,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in audited_files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
