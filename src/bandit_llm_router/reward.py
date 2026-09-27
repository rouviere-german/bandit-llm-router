"""Cost-penalized reward shaping.

The router optimizes one scalar, so the cost-quality trade-off has to be written down explicitly:

    reward = clip(quality - cost_weight * min(1, cost / cost_reference), 0, 1)

``quality`` is in [0, 1] (a rubric score offline, a rubric or user signal live) and the cost term is
the call's estimated spend relative to a fixed reference cost, capped at 1 so a single expensive
call cannot dominate. The whole reward therefore stays in [0, 1], which is what the bandit
posteriors and the importance-weighted estimators assume: bounded rewards keep the ridge noise scale
meaningful and keep IPS/DR variance finite.

``cost_weight`` is the only knob that decides how much quality a dollar is worth, and changing it
changes what "better" means. It is part of the policy version (:meth:`RewardConfig.digest_payload`),
so a policy trained under one trade-off is never compared against logs shaped under another.
MODEL_CARD.md records that the offline ``quality`` is a simulator proxy, not a measured human score.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RewardConfig", "cost_penalty", "shape_reward", "shape_rewards"]

FloatArray = npt.NDArray[np.float64]


class RewardConfig(BaseModel):
    """Weights of the reward. Frozen: it defines the objective, so it is versioned, not tuned
    live."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_reference_micros: int = Field(default=14_000, gt=0)

    def digest_payload(self) -> dict[str, object]:
        return {
            "cost_weight": self.cost_weight,
            "cost_reference_micros": self.cost_reference_micros,
        }


def cost_penalty(cost_micros: float, config: RewardConfig) -> float:
    """Penalty in [0, ``cost_weight``] for a call costing ``cost_micros`` micro-USD."""
    if cost_micros < 0:
        raise ValueError("cost_micros must be non-negative")
    return config.cost_weight * min(1.0, cost_micros / config.cost_reference_micros)


def shape_reward(quality: float, cost_micros: float, config: RewardConfig) -> float:
    """Shape one (quality, cost) pair into a reward in [0, 1]."""
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be in [0, 1]")
    return float(min(1.0, max(0.0, quality - cost_penalty(cost_micros, config))))


def shape_rewards(
    quality: npt.ArrayLike, cost_micros: npt.ArrayLike, config: RewardConfig
) -> FloatArray:
    """Vectorized :func:`shape_reward`; shapes must broadcast."""
    quality_array = np.asarray(quality, dtype=np.float64)
    cost_array = np.asarray(cost_micros, dtype=np.float64)
    if np.any(quality_array < 0.0) or np.any(quality_array > 1.0):
        raise ValueError("quality must be in [0, 1]")
    if np.any(cost_array < 0.0):
        raise ValueError("cost_micros must be non-negative")
    penalty = config.cost_weight * np.minimum(1.0, cost_array / config.cost_reference_micros)
    return np.clip(quality_array - penalty, 0.0, 1.0)
