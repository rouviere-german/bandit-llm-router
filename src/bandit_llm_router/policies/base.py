"""Action distributions, propensity floors and the policy interface.

Every routing policy in this repo returns a *distribution* over arms, never a bare arm. That is not
decoration: the logged probability of the action actually taken (its propensity) is the only thing
that makes the log reusable later. Without it, off-policy evaluation is guesswork; with it, IPS,
SNIPS and DR are all computable from data the router already writes.

Two invariants are enforced here and property-tested in ``tests/unit/test_policies.py``:

1. A distribution sums to 1 (to floating-point tolerance) and has no negative entries.
2. Every arm keeps at least ``floor`` probability. :func:`apply_propensity_floor` mixes a policy's
   own distribution with the uniform one, which bounds the importance weight of any future candidate
   by ``1 / floor`` and keeps the log informative about arms the current policy dislikes. The floor
   is the price of being able to evaluate a successor policy at all, so it is a first-class setting
   rather than an exploration afterthought.

:func:`restrict_distribution` handles the two cases where the *realized* policy differs from the one
that was sampled -- an arm whose circuit breaker is open, and an arm whose cost ceiling does not fit
the tenant's remaining budget. Mass on a blocked arm moves to the first available arm in the
fallback order, and the resulting distribution is what gets logged as the propensity, because that
is the policy that actually ran.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from bandit_llm_router.core.canonical import canonical_json, sha256_hex

__all__ = [
    "DEFAULT_PROPENSITY_FLOOR",
    "Decision",
    "NoArmAvailableError",
    "Policy",
    "PolicySnapshot",
    "apply_propensity_floor",
    "greedy_probabilities",
    "policy_version",
    "restrict_distribution",
    "sample_action",
    "sample_action_from_uniform",
    "validate_distribution",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

DEFAULT_PROPENSITY_FLOOR = 0.02
_TOLERANCE = 1e-9


class NoArmAvailableError(RuntimeError):
    """Every arm is blocked (breaker open, or no arm's cost ceiling fits the budget)."""


def validate_distribution(
    probabilities: FloatArray, *, floor: float = 0.0, tolerance: float = _TOLERANCE
) -> FloatArray:
    """Return ``probabilities`` if it is a distribution respecting ``floor``, else raise."""
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("a distribution must be a non-empty 1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("distribution contains NaN or infinite values")
    if np.any(array < -tolerance):
        raise ValueError("distribution contains negative probabilities")
    total = float(array.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"distribution sums to {total}, not 1")
    if floor > 0.0 and np.any(array < floor - 1e-9):
        raise ValueError(f"distribution violates the propensity floor {floor}")
    return array


def greedy_probabilities(values: FloatArray) -> FloatArray:
    """One-hot on the arm with the highest value; exact ties are split evenly."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain NaN or infinite entries")
    best = array == array.max()
    onehot: FloatArray = best.astype(np.float64) / float(best.sum())
    return onehot


def apply_propensity_floor(probabilities: FloatArray, floor: float) -> FloatArray:
    """Mix with the uniform distribution so every arm keeps at least ``floor`` probability.

    ``p_floored = (1 - K * floor) * p + floor``. This is exact rather than a clip-and-renormalize:
    it keeps the ordering of the original distribution and lands every arm exactly at or above the
    floor, which a clip does not guarantee.
    """
    array = np.asarray(probabilities, dtype=np.float64)
    n_arms = array.size
    if floor < 0.0:
        raise ValueError("floor must be non-negative")
    if floor * n_arms > 1.0 + _TOLERANCE:
        raise ValueError(f"floor {floor} is impossible for {n_arms} arms")
    if floor == 0.0:
        return validate_distribution(array)
    mixed = (1.0 - floor * n_arms) * array + floor
    return validate_distribution(mixed, floor=floor)


def restrict_distribution(
    probabilities: FloatArray, blocked: BoolArray, fallback_order: tuple[int, ...]
) -> FloatArray:
    """Move blocked arms' probability mass to the first available arm in ``fallback_order``.

    The result is the distribution the router *realized*, so it is what gets logged. Blocked arms
    end at exactly zero. Raises :class:`NoArmAvailableError` when nothing is available.
    """
    array = validate_distribution(np.asarray(probabilities, dtype=np.float64))
    mask = np.asarray(blocked, dtype=np.bool_)
    if mask.shape != array.shape:
        raise ValueError("blocked mask must have the same shape as the distribution")
    if bool(mask.all()):
        raise NoArmAvailableError("every arm is blocked")
    if not bool(mask.any()):
        return array
    if sorted(fallback_order) != list(range(array.size)):
        raise ValueError("fallback_order must be a permutation of the arm indices")
    substitute = {
        index: next(candidate for candidate in fallback_order if not mask[candidate])
        for index in range(array.size)
        if mask[index]
    }
    realized = np.where(mask, 0.0, array)
    for blocked_index, target in substitute.items():
        realized[target] += array[blocked_index]
    return validate_distribution(realized)


def sample_action_from_uniform(probabilities: FloatArray, uniform: float) -> int:
    """Inverse-CDF sampling from a pre-drawn uniform (no RNG call in the hot loop)."""
    if not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be in [0, 1)")
    cumulative = np.cumsum(probabilities)
    index = int(np.searchsorted(cumulative, uniform * float(cumulative[-1]), side="right"))
    return min(index, probabilities.size - 1)


def sample_action(probabilities: FloatArray, rng: np.random.Generator) -> int:
    """Draw an arm index from ``probabilities``."""
    return sample_action_from_uniform(probabilities, float(rng.random()))


@dataclass(frozen=True, slots=True)
class Decision:
    """One routing decision: the chosen arm, its propensity and the full distribution."""

    arm_index: int
    propensity: float
    probabilities: FloatArray

    def __post_init__(self) -> None:
        if not 0 <= self.arm_index < self.probabilities.size:
            raise ValueError("arm_index is outside the distribution")
        if self.propensity <= 0.0:
            raise ValueError("propensity must be positive: a zero-propensity action is unusable")


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """A policy frozen for storage: JSON params plus float arrays (no pickle anywhere)."""

    name: str
    version: str
    params: Mapping[str, JsonValue]
    arrays: Mapping[str, FloatArray]


def policy_version(name: str, payload: Mapping[str, JsonValue]) -> str:
    """Stable version string for a policy's configuration: ``<name>-<12 hex>``."""
    digest = sha256_hex(canonical_json({"name": name, **dict(payload)}))
    return f"{name}-{digest[:12]}"


@runtime_checkable
class Policy(Protocol):
    """A routing policy: distributions over arms, optional online updates, and a version."""

    name: str
    n_arms: int
    dim: int

    @property
    def version(self) -> str:
        """Version of the policy's *configuration* (hyperparameters, arms, feature version)."""
        ...

    @property
    def updates(self) -> int:
        """How many reward updates this instance has applied."""
        ...

    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Action distribution for one context. Randomized policies draw from ``rng``."""
        ...

    def probabilities_matrix(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Action distributions for a batch of contexts, shape ``(n, n_arms)``."""
        ...

    def update(
        self, features: FloatArray, arm_index: int, reward: float, weight: float = 1.0
    ) -> None:
        """Absorb one (context, action, reward) observation. No-op for static policies."""
        ...

    def snapshot(self) -> PolicySnapshot:
        """Everything needed to rebuild this policy exactly."""
        ...
