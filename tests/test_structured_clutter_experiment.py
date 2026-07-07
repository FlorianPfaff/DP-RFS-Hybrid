from dp_rfs_hybrid import run_structured_clutter_experiment


def test_structured_clutter_experiment_reports_scan_records() -> None:
    result = run_structured_clutter_experiment(scans=5, seed=3)

    assert len(result.records) == 5
    assert result.records[0].scan == 0
    assert result.records[-1].scan == 4
    assert result.as_rows()[0]["scan"] == 0


def test_adaptive_clutter_reduces_hotspot_births_in_synthetic_setup() -> None:
    result = run_structured_clutter_experiment(scans=8, seed=5)

    assert result.adaptive_final_clutter_atom_count > 0
    assert result.adaptive_total_births < result.fixed_total_births
