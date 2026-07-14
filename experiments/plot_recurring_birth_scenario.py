from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dp_rfs_hybrid import DirichletProcessBirthModel
from dp_rfs_hybrid.recurring_birth_benchmark import (
    RECURRING_BIRTH_REGIONS,
    SURVEILLANCE_BOUNDS,
    make_recurring_birth_tracker,
    simulate_recurring_birth_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--scans", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/recurring_birth_scenario.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = simulate_recurring_birth_scenario(seed=args.seed, scans=args.scans)
    tracker = make_recurring_birth_tracker("dp_delayed_recluster")
    for measurements in scenario.measurements:
        tracker.step(measurements)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    all_measurements = np.concatenate(scenario.measurements, axis=0)
    ax.scatter(
        all_measurements[:, 0],
        all_measurements[:, 1],
        s=5,
        color="#b8b8b8",
        alpha=0.28,
        linewidths=0,
        label="Measurements",
    )
    colors = ("#0072b2", "#d55e00")
    for target_index, states in enumerate(scenario.target_states):
        alive = np.isfinite(states[:, 0])
        region_index = int(scenario.target_region_indices[target_index])
        ax.plot(
            states[alive, 0],
            states[alive, 1],
            color=colors[region_index],
            linewidth=1.2,
            alpha=0.8,
        )
    ax.scatter(
        RECURRING_BIRTH_REGIONS[:, 0],
        RECURRING_BIRTH_REGIONS[:, 1],
        marker="X",
        s=90,
        color="#000000",
        label="True birth regions",
        zorder=5,
    )
    if isinstance(tracker.birth_model, DirichletProcessBirthModel):
        recurrent_atoms = [
            atom for atom in tracker.birth_model.atoms if atom.count >= 1.5
        ]
        atom_positions = np.asarray(
            [atom.state.mean[:2] for atom in recurrent_atoms]
        )
        atom_counts = np.asarray([atom.count for atom in recurrent_atoms])
        if len(atom_positions):
            ax.scatter(
                atom_positions[:, 0],
                atom_positions[:, 1],
                marker="o",
                s=45 + 35 * atom_counts,
                facecolors="none",
                edgecolors="#009e73",
                linewidths=1.8,
                label="Recurrent DP atoms",
                zorder=4,
            )
    ax.set_xlim(SURVEILLANCE_BOUNDS[0])
    ax.set_ylim(SURVEILLANCE_BOUNDS[1])
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
