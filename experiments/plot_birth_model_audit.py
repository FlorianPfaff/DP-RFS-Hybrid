"""Plot deterministic audit diagnostics, not tracking-performance results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    rows = data["posterior_feedback"]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.18, [r["likelihood_plus_minus_ratio"] for r in rows],
                width=0.36, color="#167A72", label="Likelihood evidence")
    axes[0].bar(x + 0.18, [r["overlap_plus_minus_ratio"] for r in rows],
                width=0.36, color="#B35735", label="Current posterior overlap")
    axes[0].set_xticks(x, ["Prior mean 0; observation 2", "Prior mean 2; observation 0"])
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Evidence ratio: region +2 / region -2")
    axes[0].set_title("Same track posterior, different evidence")
    axes[0].legend(loc="upper right", fontsize=8)
    rows = data["diffuse_evidence"]
    variances = [r["observation_variance"] for r in rows]
    merged = [r["merged_region_variance"] for r in rows]
    axes[1].loglog(variances, merged, "o-", color="#B35735", label="Moment-merged variance")
    axes[1].axhline(1.0, color="#167A72", linestyle="--", label="Initial region variance")
    axes[1].set_xlabel("Independent observation variance R")
    axes[1].set_ylabel("Reusable region variance")
    axes[1].set_title("Noisy-observation interpretation only")
    axes[1].legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        output = args.output_prefix.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(output)
        figure.savefig(output, dpi=180)
        print(f"wrote {output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
