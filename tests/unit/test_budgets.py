"""Per-tenant spend caps, enforced as an invariant rather than as an aspiration.

The property test is the point of the module: over any random sequence of reservations, commits and
releases, a tenant's committed spend never exceeds its cap. That holds because a call is admitted
only if its *worst case* fits, and a commit can never charge more than was reserved. Every other
test here isolates one way that invariant could be lost -- a window rolling over mid-call, a failed
call not refunding, two workers each spending the full amount.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bandit_llm_router.budgets import (
    BudgetExhaustedError,
    TenantBudgetError,
    TenantBudgetGuard,
    TenantBudgets,
    TenantSpend,
)

pytestmark = pytest.mark.unit

MICROS_PER_USD = 1_000_000


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def guard(
    *, cap_usd: float = 1.0, path: Path | None = None, window: float = 86_400.0
) -> tuple[TenantBudgetGuard, FakeClock]:
    clock = FakeClock()
    budgets = TenantBudgets(default_usd=cap_usd, window_seconds=window)
    return TenantBudgetGuard(budgets, path=path, clock=clock), clock


# --- the invariant -------------------------------------------------------------------------------


@given(
    operations=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=400_000),  # ceiling to reserve
            st.integers(min_value=0, max_value=500_000),  # actual cost, may overshoot
            st.booleans(),  # True commits, False releases
        ),
        min_size=1,
        max_size=40,
    ),
    cap_micros=st.integers(min_value=1, max_value=2_000_000),
)
def test_committed_spend_never_exceeds_the_cap(
    operations: list[tuple[int, int, bool]], cap_micros: int
) -> None:
    keeper = TenantBudgetGuard(
        TenantBudgets(default_usd=cap_micros / MICROS_PER_USD), clock=FakeClock()
    )
    for ceiling, actual, commit in operations:
        try:
            reservation = keeper.reserve("tenant_a", "haiku", ceiling)
        except BudgetExhaustedError:
            continue
        if commit:
            charged = keeper.commit(reservation, actual)
            assert charged <= ceiling
        else:
            keeper.release(reservation)
        assert keeper.spent_micros("tenant_a") <= cap_micros
        assert keeper.remaining_micros("tenant_a") >= 0


def test_a_commit_never_charges_more_than_was_reserved() -> None:
    """The overshoot is capped at one call, which is what makes the cap exact."""
    keeper, _ = guard(cap_usd=0.1)
    reservation = keeper.reserve("tenant_a", "haiku", 50_000)
    assert keeper.commit(reservation, 999_999) == 50_000
    assert keeper.spent_micros("tenant_a") == 50_000


def test_a_call_whose_worst_case_does_not_fit_is_refused_before_it_happens() -> None:
    keeper, _ = guard(cap_usd=0.1)
    keeper.commit(keeper.reserve("tenant_a", "haiku", 90_000), 90_000)
    with pytest.raises(BudgetExhaustedError) as failure:
        keeper.reserve("tenant_a", "opus", 20_000)
    assert failure.value.tenant == "tenant_a"
    assert failure.value.requested_micros == 20_000
    assert failure.value.remaining_micros == 10_000


def test_a_failed_call_refunds_its_reservation() -> None:
    keeper, _ = guard(cap_usd=0.1)
    reservation = keeper.reserve("tenant_a", "opus", 100_000)
    assert keeper.remaining_micros("tenant_a") == 0
    keeper.release(reservation)
    assert keeper.remaining_micros("tenant_a") == 100_000
    assert keeper.spent_micros("tenant_a") == 0


def test_reservations_are_held_against_concurrent_calls() -> None:
    keeper, _ = guard(cap_usd=0.1)
    first = keeper.reserve("tenant_a", "haiku", 60_000)
    with pytest.raises(BudgetExhaustedError):
        keeper.reserve("tenant_a", "haiku", 60_000)
    keeper.commit(first, 10_000)
    # Releasing the held amount frees room for the second call.
    assert keeper.reserve("tenant_a", "haiku", 60_000).micros == 60_000


def test_a_negative_amount_is_refused() -> None:
    keeper, _ = guard()
    with pytest.raises(ValueError, match="non-negative"):
        keeper.reserve("tenant_a", "haiku", -1)
    reservation = keeper.reserve("tenant_a", "haiku", 10)
    with pytest.raises(ValueError, match="non-negative"):
        keeper.commit(reservation, -1)


# --- caps and windows ----------------------------------------------------------------------------


def test_named_tenants_override_the_default_cap() -> None:
    budgets = TenantBudgets(caps_usd={"tenant_big": 10.0}, default_usd=1.0)
    keeper = TenantBudgetGuard(budgets, clock=FakeClock())
    assert keeper.remaining_micros("tenant_big") == 10 * MICROS_PER_USD
    assert keeper.remaining_micros("tenant_small") == MICROS_PER_USD


def test_a_negative_cap_is_a_configuration_error() -> None:
    budgets = TenantBudgets(caps_usd={"tenant_a": -1.0})
    with pytest.raises(TenantBudgetError, match="negative cap"):
        budgets.cap_micros("tenant_a")


def test_a_zero_cap_refuses_every_call() -> None:
    keeper, _ = guard(cap_usd=0.0)
    with pytest.raises(BudgetExhaustedError):
        keeper.reserve("tenant_a", "haiku", 1)


def test_the_window_rolls_over_and_resets_the_spend() -> None:
    keeper, clock = guard(cap_usd=0.1, window=100.0)
    keeper.commit(keeper.reserve("tenant_a", "haiku", 100_000), 100_000)
    assert keeper.remaining_micros("tenant_a") == 0
    clock.now += 101.0
    assert keeper.remaining_micros("tenant_a") == 100_000
    assert keeper.spent_micros("tenant_a") == 0


def test_a_window_that_rolls_mid_call_still_charges_the_call() -> None:
    """A rollover must not turn an in-flight call into free money."""
    keeper, clock = guard(cap_usd=1.0, window=100.0)
    reservation = keeper.reserve("tenant_a", "opus", 40_000)
    clock.now += 200.0
    charged = keeper.commit(reservation, 30_000)
    assert charged == 30_000
    assert keeper.spent_micros("tenant_a") == 30_000
    snapshot = keeper.snapshot()["tenant_a"]
    assert snapshot["reserved_micros"] == 0


# --- masks ---------------------------------------------------------------------------------------


def test_the_affordable_mask_reports_which_arms_still_fit() -> None:
    keeper, _ = guard(cap_usd=0.05)
    ceilings = np.array([10_000.0, 40_000.0, 100_000.0])
    assert list(keeper.affordable_mask("tenant_a", ceilings)) == [True, True, False]
    keeper.commit(keeper.reserve("tenant_a", "haiku", 45_000), 45_000)
    assert list(keeper.affordable_mask("tenant_a", ceilings)) == [False, False, False]


def test_the_snapshot_reports_every_tenant_the_guard_has_seen() -> None:
    keeper, _ = guard(cap_usd=0.2)
    keeper.commit(keeper.reserve("tenant_a", "haiku", 50_000), 50_000)
    keeper.reserve("tenant_b", "opus", 30_000)
    snapshot = keeper.snapshot()
    assert set(snapshot) == {"tenant_a", "tenant_b"}
    assert snapshot["tenant_a"]["spent_micros"] == 50_000
    assert snapshot["tenant_a"]["remaining_micros"] == 150_000
    assert snapshot["tenant_b"]["reserved_micros"] == 30_000
    assert snapshot["tenant_b"]["cap_micros"] == 200_000


# --- persistence ---------------------------------------------------------------------------------


def test_two_workers_share_one_budget_through_the_state_file(tmp_path: Path) -> None:
    """Without the shared file, every worker would happily spend the whole cap."""
    path = tmp_path / "budgets.json"
    budgets = TenantBudgets(default_usd=0.1)
    clock = FakeClock()
    worker_a = TenantBudgetGuard(budgets, path=path, clock=clock)
    worker_b = TenantBudgetGuard(budgets, path=path, clock=clock)
    worker_a.commit(worker_a.reserve("tenant_a", "haiku", 80_000), 80_000)
    assert worker_b.remaining_micros("tenant_a") == 20_000
    with pytest.raises(BudgetExhaustedError):
        worker_b.reserve("tenant_a", "opus", 50_000)


def test_the_state_file_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "budgets.json"
    budgets = TenantBudgets(default_usd=0.1)
    clock = FakeClock()
    first = TenantBudgetGuard(budgets, path=path, clock=clock)
    first.commit(first.reserve("tenant_a", "haiku", 25_000), 25_000)
    restarted = TenantBudgetGuard(budgets, path=path, clock=clock)
    assert restarted.spent_micros("tenant_a") == 25_000


def test_a_corrupt_state_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "budgets.json"
    path.write_text('{"version": 1, "tenants": {"tenant_a": {"nonsense": true}}}', encoding="utf-8")
    with pytest.raises(TenantBudgetError, match="failed validation"):
        TenantBudgetGuard(TenantBudgets(), path=path, clock=FakeClock())


def test_an_absent_state_file_starts_empty(tmp_path: Path) -> None:
    keeper = TenantBudgetGuard(TenantBudgets(), path=tmp_path / "absent.json", clock=FakeClock())
    assert keeper.snapshot() == {}


def test_the_spend_record_refuses_negative_amounts() -> None:
    TenantSpend(window_start=0.0, spent_micros=0, reserved_micros=0)
    with pytest.raises(ValidationError):
        TenantSpend(window_start=0.0, spent_micros=-1)
    with pytest.raises(ValidationError):
        TenantSpend(window_start=0.0, reserved_micros=-1)


def test_the_budget_configuration_is_frozen_and_versioned() -> None:
    budgets = TenantBudgets(caps_usd={"b": 2.0, "a": 1.0})
    payload = budgets.digest_payload()
    caps = payload["caps_usd"]
    assert isinstance(caps, dict)
    assert list(caps) == ["a", "b"]  # sorted, so the digest is stable
    with pytest.raises(ValueError, match="frozen"):
        budgets.default_usd = 5.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TenantBudgets(window_seconds=0.0)
