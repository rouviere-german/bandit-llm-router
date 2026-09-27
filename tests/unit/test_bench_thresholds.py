"""The bench scripts' threshold arithmetic, which nothing else covers.

`bench/thresholds.json` is what CI fails on, so a bug in how a bench turns a result into a number is
a bug in the safety net itself -- and it fails *silently*, by passing or failing for the wrong
reason. One such bug existed here: `float(recovery.recovery_requests or SENTINEL)` mapped the best
possible adaptation outcome ("the shift never pushed this policy below target", reported as `0`) to
the sentinel for the worst one, because `0` is falsy. It turned a clean pass into a regression
report. These tests pin the three-way distinction so it cannot come back.

The bench modules are scripts, not part of the package, so they are loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from bandit_llm_router.experiments import ShiftRecovery

pytestmark = pytest.mark.unit

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"


def _load(name: str) -> ModuleType:
    """Import a bench script by path, with bench/ on sys.path for its `_common` import."""
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", BENCH_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_router() -> ModuleType:
    return _load("bench_router")


def recovery(
    *, dipped: bool, requests: int | None, pre: float = 0.99, trough: float = 0.60
) -> ShiftRecovery:
    return ShiftRecovery(
        policy_name="p",
        discount=1.0,
        pre_shift_ratio=pre,
        post_shift_trough=trough,
        recovery_requests=requests,
        final_ratio=0.99,
        dipped_below_target=dipped,
    )


def test_never_dipping_is_the_best_outcome_and_scores_zero(bench_router: ModuleType) -> None:
    """The regression this file exists for: 0 must not be read as the sentinel."""
    best = recovery(dipped=False, requests=0, trough=0.98)
    assert bench_router._recovery_cost(best) == 0.0
    assert bench_router._never_recovered(best) == 0.0


def test_recovering_after_k_requests_scores_k(bench_router: ModuleType) -> None:
    assert bench_router._recovery_cost(recovery(dipped=True, requests=19_851)) == 19_851.0


def test_recovering_immediately_scores_zero_not_the_sentinel(bench_router: ModuleType) -> None:
    """A policy that dipped and bounced back inside one step also reports 0 requests."""
    assert bench_router._recovery_cost(recovery(dipped=True, requests=0)) == 0.0


def test_dipping_and_staying_down_is_the_worst_outcome(bench_router: ModuleType) -> None:
    worst = recovery(dipped=True, requests=None)
    assert bench_router._recovery_cost(worst) == bench_router._NEVER_RECOVERED
    assert bench_router._never_recovered(worst) == 1.0


def test_the_three_outcomes_are_strictly_ordered(bench_router: ModuleType) -> None:
    never_dipped = bench_router._recovery_cost(recovery(dipped=False, requests=0))
    recovered = bench_router._recovery_cost(recovery(dipped=True, requests=5_000))
    stuck = bench_router._recovery_cost(recovery(dipped=True, requests=None))
    assert never_dipped < recovered < stuck


def test_the_sentinel_is_far_above_any_plausible_recovery(bench_router: ModuleType) -> None:
    """So it can only ever read as a failure against a threshold in requests."""
    assert bench_router._NEVER_RECOVERED > 1_000 * bench_router.DEFAULT_SHIFT_REQUESTS


def test_a_toothless_shift_is_reported_as_a_shortfall(bench_router: ModuleType) -> None:
    """Guards the guard: if the stationary policy stops being hurt, the comparison is vacuous."""
    depth = bench_router._MIN_STATIONARY_TROUGH_DEPTH
    assert 0.0 < depth < 1.0
    biting = recovery(dipped=True, requests=1, pre=0.99, trough=0.99 - depth - 0.05)
    toothless = recovery(dipped=True, requests=1, pre=0.99, trough=0.99 - depth / 2)
    assert max(0.0, depth - (biting.pre_shift_ratio - biting.post_shift_trough)) == 0.0
    assert max(0.0, depth - (toothless.pre_shift_ratio - toothless.post_shift_trough)) > 0.0


def test_every_threshold_section_matches_a_bench_script() -> None:
    """A section nobody writes to would silently never be checked."""
    import json

    document = json.loads((BENCH_DIR / "thresholds.json").read_text(encoding="utf-8"))
    sections = {key for key in document if not key.startswith("_")}
    scripts = {path.stem for path in BENCH_DIR.glob("bench_*.py")}
    assert sections == scripts


def test_thresholds_are_enforced_in_the_same_direction() -> None:
    """Every threshold is a numeric upper bound: no negative limits, no strings, no nulls."""
    import json

    document = json.loads((BENCH_DIR / "thresholds.json").read_text(encoding="utf-8"))
    for section, benchmarks in document.items():
        if section.startswith("_"):
            continue
        for name, limits in benchmarks.items():
            for metric, limit in limits.items():
                assert isinstance(limit, (int, float)), f"{section}.{name}.{metric}"
                assert not isinstance(limit, bool), f"{section}.{name}.{metric}"
                assert float(limit) >= 0.0, f"{section}.{name}.{metric}"
