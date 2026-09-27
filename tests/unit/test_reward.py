"""Cost-penalized reward shaping: the one place where a dollar is converted into quality.

The reward has to stay in [0, 1] for two downstream reasons, and the property tests check both: the
ridge posteriors assume a bounded target, and IPS/DR variance is only finite for bounded rewards.
The reward configuration is also part of the policy version, so a policy trained under one
trade-off can never be compared against logs shaped under another.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bandit_llm_router.reward import RewardConfig, cost_penalty, shape_reward, shape_rewards

pytestmark = pytest.mark.unit

qualities = st.floats(0.0, 1.0, allow_nan=False, width=64)
costs = st.floats(0.0, 1e7, allow_nan=False, allow_infinity=False, width=64)


@given(qualities, costs)
def test_the_reward_always_stays_in_the_unit_interval(quality: float, cost: float) -> None:
    assert 0.0 <= shape_reward(quality, cost, RewardConfig()) <= 1.0


@given(qualities, costs, st.floats(0.0, 1.0, allow_nan=False, width=64))
def test_the_bound_holds_for_every_cost_weight(quality: float, cost: float, weight: float) -> None:
    config = RewardConfig(cost_weight=weight)
    assert 0.0 <= shape_reward(quality, cost, config) <= 1.0


@given(qualities, costs)
def test_the_penalty_never_exceeds_the_cost_weight(quality: float, cost: float) -> None:
    config = RewardConfig(cost_weight=0.4)
    penalty = cost_penalty(cost, config)
    assert 0.0 <= penalty <= config.cost_weight
    assert shape_reward(quality, cost, config) == pytest.approx(
        max(0.0, min(1.0, quality - penalty))
    )


@given(costs, costs)
def test_the_penalty_is_monotone_in_cost(cheap: float, dear: float) -> None:
    config = RewardConfig()
    low, high = sorted((cheap, dear))
    assert cost_penalty(low, config) <= cost_penalty(high, config) + 1e-12


def test_a_single_expensive_call_cannot_dominate_the_objective() -> None:
    """The penalty saturates: a 1000x overspend costs the same as a 1x reference overspend."""
    config = RewardConfig(cost_weight=0.5, cost_reference_micros=10_000)
    assert cost_penalty(10_000, config) == pytest.approx(0.5)
    assert cost_penalty(10_000_000, config) == pytest.approx(0.5)


def test_a_zero_cost_weight_makes_the_reward_pure_quality() -> None:
    config = RewardConfig(cost_weight=0.0)
    assert shape_reward(0.7, 10_000_000, config) == pytest.approx(0.7)


def test_shaping_validates_its_inputs() -> None:
    config = RewardConfig()
    with pytest.raises(ValueError, match="quality must be in"):
        shape_reward(1.5, 0.0, config)
    with pytest.raises(ValueError, match="non-negative"):
        shape_reward(0.5, -1.0, config)
    with pytest.raises(ValueError, match="non-negative"):
        cost_penalty(-1.0, config)


def test_the_vectorized_form_matches_the_scalar_one() -> None:
    config = RewardConfig(cost_weight=0.3, cost_reference_micros=8_000)
    quality = np.array([[0.1, 0.9], [0.5, 0.2]])
    cost = np.array([[1_000.0, 20_000.0], [8_000.0, 0.0]])
    shaped = shape_rewards(quality, cost, config)
    assert shaped.shape == quality.shape
    for row in range(2):
        for column in range(2):
            assert shaped[row, column] == pytest.approx(
                shape_reward(quality[row, column], cost[row, column], config)
            )


def test_the_vectorized_form_validates_its_inputs() -> None:
    config = RewardConfig()
    with pytest.raises(ValueError, match="quality must be in"):
        shape_rewards(np.array([1.5]), np.array([0.0]), config)
    with pytest.raises(ValueError, match="non-negative"):
        shape_rewards(np.array([0.5]), np.array([-1.0]), config)


def test_the_configuration_is_frozen_and_part_of_a_policy_version() -> None:
    config = RewardConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.cost_weight = 0.9  # type: ignore[misc]
    assert config.digest_payload() == {
        "cost_weight": config.cost_weight,
        "cost_reference_micros": config.cost_reference_micros,
    }
    assert RewardConfig(cost_weight=0.6).digest_payload() != config.digest_payload()


def test_the_configuration_bounds_its_own_fields() -> None:
    with pytest.raises(ValidationError):
        RewardConfig(cost_weight=1.5)
    with pytest.raises(ValidationError):
        RewardConfig(cost_reference_micros=0)


def test_cheaper_wins_when_quality_is_tied() -> None:
    """The reason routing pays off at all: on easy requests the arms tie and cost decides."""
    config = RewardConfig()
    assert shape_reward(0.9, 2_000, config) > shape_reward(0.9, 20_000, config)


def test_quality_wins_when_the_gap_is_worth_the_money() -> None:
    config = RewardConfig(cost_weight=0.5, cost_reference_micros=14_000)
    cheap_but_weak = shape_reward(0.45, 2_000, config)
    dear_but_strong = shape_reward(0.95, 14_000, config)
    assert dear_but_strong > cheap_but_weak
