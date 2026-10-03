import numpy as np
import pytest

from dp_rfs_hybrid.coherent_analysis import paired_summary, verdict


def test_paired_block_bootstrap_and_failure_rules():
    base = np.arange(1, 11)[:, None]*np.ones((1, 3))
    summary = paired_summary(base*.96, base)
    np.testing.assert_allclose(summary["percent_ci"], [4., 4.])
    assert verdict(summary, summary, [summary, summary]) == "continue"
    assert verdict(summary, summary, [], failures=1) == "inconclusive"
    worse = paired_summary(base*1.06, base)
    assert verdict(summary, summary, [worse]) == "no-go"
    tie = paired_summary(base, base)
    assert verdict(tie, tie, [tie]) == "no-go"
    with pytest.raises(ValueError):
        paired_summary([[float("nan")]], [[1.]])
