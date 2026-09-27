"""The per-arm circuit breaker, and the propensity it forces the router to log.

The breaker itself is small. What matters for this repo is the second half of these tests: when an
arm is unavailable the request is served by a substitute, so the distribution that actually ran is
*not* the one the policy returned. Logging the policy's propensity there would poison every later
off-policy estimate, so the realized distribution is computed and logged instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from bandit_llm_router.breaker import BreakerConfig, BreakerState, CircuitBreaker
from bandit_llm_router.policies.base import restrict_distribution

pytestmark = pytest.mark.unit

ARMS = ("haiku", "sonnet", "opus")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def breaker(**overrides: float | int) -> tuple[CircuitBreaker, FakeClock]:
    clock = FakeClock()
    config = BreakerConfig.model_validate(
        {"failure_threshold": 3, "cooldown_seconds": 30.0, **overrides}
    )
    return CircuitBreaker(ARMS, config=config, clock=clock), clock


# --- state transitions ---------------------------------------------------------------------------


def test_a_healthy_arm_starts_closed_and_available() -> None:
    circuit, _ = breaker()
    assert circuit.state("haiku") is BreakerState.CLOSED
    assert circuit.is_available("haiku")
    assert circuit.snapshot() == dict.fromkeys(ARMS, "closed")
    assert circuit.any_available()
    assert circuit.arms == ARMS
    assert circuit.config.failure_threshold == 3


def test_consecutive_failures_open_the_arm() -> None:
    circuit, _ = breaker(failure_threshold=3)
    for _ in range(2):
        circuit.record_failure("haiku")
    assert circuit.state("haiku") is BreakerState.CLOSED
    circuit.record_failure("haiku")
    assert circuit.state("haiku") is BreakerState.OPEN
    assert not circuit.is_available("haiku")
    # Only that arm is affected.
    assert circuit.state("sonnet") is BreakerState.CLOSED


def test_a_success_resets_the_failure_count() -> None:
    circuit, _ = breaker(failure_threshold=3)
    circuit.record_failure("haiku")
    circuit.record_failure("haiku")
    circuit.record_success("haiku")
    circuit.record_failure("haiku")
    circuit.record_failure("haiku")
    assert circuit.state("haiku") is BreakerState.CLOSED


def test_the_arm_becomes_half_open_after_the_cooldown() -> None:
    circuit, clock = breaker(failure_threshold=1, cooldown_seconds=30.0)
    circuit.record_failure("haiku")
    assert circuit.state("haiku") is BreakerState.OPEN
    clock.now += 29.0
    assert circuit.state("haiku") is BreakerState.OPEN
    clock.now += 2.0
    assert circuit.state("haiku") is BreakerState.HALF_OPEN
    # Half-open counts as available: one trial request is how the arm recovers.
    assert circuit.is_available("haiku")
    assert not bool(circuit.blocked_mask()[0])


def test_a_successful_trial_closes_the_arm() -> None:
    circuit, clock = breaker(failure_threshold=1, cooldown_seconds=10.0)
    circuit.record_failure("haiku")
    clock.now += 11.0
    circuit.record_success("haiku")
    assert circuit.state("haiku") is BreakerState.CLOSED


def test_a_failed_trial_opens_the_arm_again_for_a_full_cooldown() -> None:
    circuit, clock = breaker(failure_threshold=1, cooldown_seconds=10.0)
    circuit.record_failure("haiku")
    clock.now += 11.0
    assert circuit.state("haiku") is BreakerState.HALF_OPEN
    circuit.record_failure("haiku")
    assert circuit.state("haiku") is BreakerState.OPEN
    clock.now += 9.0
    assert circuit.state("haiku") is BreakerState.OPEN
    clock.now += 2.0
    assert circuit.state("haiku") is BreakerState.HALF_OPEN


def test_several_trial_successes_can_be_demanded() -> None:
    circuit, clock = breaker(failure_threshold=1, cooldown_seconds=5.0, half_open_successes=2)
    circuit.record_failure("opus")
    clock.now += 6.0
    circuit.record_success("opus")
    assert circuit.state("opus") is BreakerState.HALF_OPEN
    circuit.record_success("opus")
    assert circuit.state("opus") is BreakerState.CLOSED


def test_the_blocked_mask_follows_the_arm_order() -> None:
    circuit, _ = breaker(failure_threshold=1)
    circuit.record_failure("sonnet")
    mask = circuit.blocked_mask()
    assert list(mask) == [False, True, False]
    assert circuit.snapshot()["sonnet"] == "open"
    assert circuit.any_available()


def test_every_arm_open_means_nothing_is_available() -> None:
    circuit, _ = breaker(failure_threshold=1)
    for arm in ARMS:
        circuit.record_failure(arm)
    assert list(circuit.blocked_mask()) == [True, True, True]
    assert not circuit.any_available()


def test_an_unknown_arm_is_rejected() -> None:
    circuit, _ = breaker()
    for call in (circuit.state, circuit.record_success, circuit.record_failure):
        with pytest.raises(ValueError, match="unknown arm"):
            call("gpt")


def test_the_breaker_needs_at_least_one_arm() -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        CircuitBreaker([])


def test_the_configuration_is_validated() -> None:
    with pytest.raises(ValidationError):
        BreakerConfig(failure_threshold=0)
    with pytest.raises(ValidationError):
        BreakerConfig(cooldown_seconds=0.0)
    with pytest.raises(ValidationError):
        BreakerConfig(half_open_successes=0)


# --- what the breaker does to a logged propensity ------------------------------------------------


def test_the_logged_propensity_is_the_realized_one_not_the_policys() -> None:
    """The whole reason the breaker lives next to the policy code."""
    circuit, _ = breaker(failure_threshold=1)
    circuit.record_failure("opus")
    policy_distribution = np.array([0.2, 0.3, 0.5])
    realized = restrict_distribution(policy_distribution, circuit.blocked_mask(), (0, 1, 2))
    # Opus's half of the traffic moved to the cheapest available arm in the fallback order.
    assert realized == pytest.approx([0.7, 0.3, 0.0])
    assert realized.sum() == pytest.approx(1.0)
    # Logging 0.5 for a haiku-served request would overstate the propensity by 40%.
    assert realized[0] != policy_distribution[0]


def test_a_breaker_adjusted_row_is_still_a_usable_log_row() -> None:
    circuit, _ = breaker(failure_threshold=1)
    circuit.record_failure("haiku")
    realized = restrict_distribution(np.array([0.6, 0.3, 0.1]), circuit.blocked_mask(), (0, 1, 2))
    served = int(np.argmax(realized))
    # Positive propensity for the arm that served it: the row can be reweighted later.
    assert realized[served] > 0.0
    assert realized[0] == 0.0
