"""Context drift monitoring over the logged feature columns.

A learned router acts confidently on contexts it has no evidence about when the traffic changes, and
the off-policy estimates that screened it were computed on a distribution that no longer exists.
Input drift needs no labels, so it is the earliest available signal; these tests check the monitor
actually fires on a shifted column and stays quiet on a stable one.
"""

from __future__ import annotations

import numpy as np
import pytest

from bandit_llm_router.features import DENSE_FEATURE_NAMES, HASH_BUCKETS, Featurizer
from bandit_llm_router.monitoring import ContextDriftMonitor

pytestmark = pytest.mark.unit

NAMES = ("length", "digits", "bias")


def reference(seed: int = 0, rows: int = 2_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.normal(0.4, 0.1, size=rows), rng.uniform(0.0, 0.2, size=rows), np.ones(rows)]
    )


def test_a_stable_window_reports_no_drift() -> None:
    monitor = ContextDriftMonitor(reference(0), NAMES)
    scores = monitor.psi_by_feature(reference(1))
    assert set(scores) == set(NAMES)
    assert all(score < 0.1 for score in scores.values())
    assert monitor.drifted_features(reference(1)) == ()


def test_a_shifted_column_is_named() -> None:
    """The single case the monitor exists for: a client starts sending much longer inputs."""
    current = reference(2)
    current[:, 0] += 0.5
    monitor = ContextDriftMonitor(reference(0), NAMES)
    scores = monitor.psi_by_feature(current)
    assert scores["length"] > 0.2
    assert scores["digits"] < 0.1
    assert monitor.drifted_features(current) == ("length",)


def test_a_constant_column_reports_zero_by_construction() -> None:
    monitor = ContextDriftMonitor(reference(0), NAMES)
    assert monitor.psi_by_feature(reference(3))["bias"] == pytest.approx(0.0)


def test_the_threshold_decides_what_counts_as_drift() -> None:
    current = reference(4)
    current[:, 0] += 0.04
    measured = ContextDriftMonitor(reference(0), NAMES).psi_by_feature(current)["length"]
    assert 0.0 < measured < 10.0
    lenient = ContextDriftMonitor(reference(0), NAMES, psi_threshold=measured * 2.0)
    strict = ContextDriftMonitor(reference(0), NAMES, psi_threshold=measured / 2.0)
    assert lenient.drifted_features(current) == ()
    assert "length" in strict.drifted_features(current)


def test_the_full_report_adds_the_ks_test() -> None:
    current = reference(5)
    current[:, 1] += 0.4
    monitor = ContextDriftMonitor(reference(0), NAMES)
    report = monitor.check(current)
    assert report.drifted
    assert "digits" in report.drifted_features
    assert monitor.feature_names == NAMES


def test_the_monitor_validates_its_reference() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        ContextDriftMonitor(np.zeros(5), ("a",))
    with pytest.raises(ValueError, match="2-D array"):
        ContextDriftMonitor(np.zeros((1, 3)), NAMES)
    with pytest.raises(ValueError, match="must name every reference column"):
        ContextDriftMonitor(reference(0), ("a", "b"))
    with pytest.raises(ValueError, match="NaN"):
        ContextDriftMonitor(np.full((10, 3), np.nan), NAMES)
    with pytest.raises(ValueError, match="psi_threshold"):
        ContextDriftMonitor(reference(0), NAMES, psi_threshold=0.0)


def test_the_monitor_validates_the_window_it_is_given() -> None:
    monitor = ContextDriftMonitor(reference(0), NAMES)
    with pytest.raises(ValueError, match="same number of columns"):
        monitor.psi_by_feature(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least two rows"):
        monitor.psi_by_feature(np.zeros((1, 3)))


def test_the_monitor_names_the_routers_real_feature_columns(featurizer: Featurizer) -> None:
    """The gauge is labelled by feature name, so the names must be the featurizer's own."""
    names = featurizer.feature_names()
    assert len(names) == HASH_BUCKETS + len(DENSE_FEATURE_NAMES) + 1
    reference_rows = np.random.default_rng(0).normal(0.0, 0.2, size=(200, len(names)))
    monitor = ContextDriftMonitor(reference_rows, names)
    scores = monitor.psi_by_feature(reference_rows + 0.01)
    assert set(scores) == set(names)
