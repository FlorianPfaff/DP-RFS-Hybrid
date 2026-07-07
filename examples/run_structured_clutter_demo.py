from __future__ import annotations

from dp_rfs_hybrid import run_structured_clutter_experiment


def main() -> None:
    result = run_structured_clutter_experiment(scans=20, seed=11)

    print("scan measurements fixed_births adaptive_births fixed_tracks adaptive_tracks clutter_atoms")
    for record in result.records:
        print(
            f"{record.scan:02d} {record.measurement_count:12d} "
            f"{record.fixed_birth_count:12d} {record.adaptive_birth_count:15d} "
            f"{record.fixed_active_track_count:12d} {record.adaptive_active_track_count:15d} "
            f"{record.adaptive_clutter_atom_count:13d}"
        )

    print("\nSummary:")
    print(f"  fixed total births:    {result.fixed_total_births}")
    print(f"  adaptive total births: {result.adaptive_total_births}")
    print(f"  fixed final tracks:    {result.fixed_final_track_count}")
    print(f"  adaptive final tracks: {result.adaptive_final_track_count}")
    print(f"  clutter atoms:         {result.adaptive_final_clutter_atom_count}")

    if result.adaptive_clutter_atom_means:
        print("\nLearned clutter atom means:")
        for index, mean in enumerate(result.adaptive_clutter_atom_means, start=1):
            print(f"  atom {index}: ({mean[0]:.2f}, {mean[1]:.2f})")


if __name__ == "__main__":
    main()
