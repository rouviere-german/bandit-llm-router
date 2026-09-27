"""Per-arm circuit breaker, and what it does to a logged propensity.

When an arm's provider starts failing, routing to it wastes money and latency. The breaker counts
consecutive failures per arm, opens after ``failure_threshold``, and refuses that arm for
``cooldown_seconds``; then it lets exactly one trial through (half-open) and either closes or opens
again.

The part that matters for this repo is what happens to the *log*. If the policy sampled an arm whose
breaker is open, the request is served by the first available arm in the catalog's fallback order --
so the action actually taken was drawn from a different distribution than the policy's. Logging the
policy's propensity would be a lie, and every later off-policy estimate built on that log would
inherit it. Instead the realized distribution is computed by moving each blocked arm's mass onto its
substitute (:func:`~bandit_llm_router.policies.base.restrict_distribution`) and *that* propensity is
logged, together with a flag saying the row was breaker-adjusted. The same mechanism handles arms
whose cost ceiling does not fit the tenant's remaining budget.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BreakerConfig", "BreakerState", "CircuitBreaker"]

BoolArray = npt.NDArray[np.bool_]


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerConfig(BaseModel):
    """Breaker thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: float = Field(default=30.0, gt=0.0)
    half_open_successes: int = Field(default=1, ge=1)


class CircuitBreaker:
    """Thread-safe per-arm breaker over a fixed arm set."""

    def __init__(
        self,
        arms: Sequence[str],
        *,
        config: BreakerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not arms:
            raise ValueError("at least one arm is required")
        self._arms = tuple(arms)
        self._config = config if config is not None else BreakerConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = dict.fromkeys(self._arms, 0)
        self._successes = dict.fromkeys(self._arms, 0)
        self._opened_at: dict[str, float] = {}

    @property
    def arms(self) -> tuple[str, ...]:
        return self._arms

    @property
    def config(self) -> BreakerConfig:
        return self._config

    def _state(self, arm: str) -> BreakerState:
        opened = self._opened_at.get(arm)
        if opened is None:
            return BreakerState.CLOSED
        if self._clock() - opened >= self._config.cooldown_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def state(self, arm: str) -> BreakerState:
        if arm not in self._failures:
            raise ValueError(f"unknown arm {arm!r}")
        with self._lock:
            return self._state(arm)

    def is_available(self, arm: str) -> bool:
        """Half-open counts as available: one trial request is how it recovers."""
        return self.state(arm) is not BreakerState.OPEN

    def blocked_mask(self) -> BoolArray:
        """Boolean mask over the arm order: True where the arm must not be used."""
        with self._lock:
            return np.asarray(
                [self._state(arm) is BreakerState.OPEN for arm in self._arms], dtype=np.bool_
            )

    def record_success(self, arm: str) -> None:
        if arm not in self._failures:
            raise ValueError(f"unknown arm {arm!r}")
        with self._lock:
            if self._state(arm) is BreakerState.HALF_OPEN:
                self._successes[arm] += 1
                if self._successes[arm] >= self._config.half_open_successes:
                    self._opened_at.pop(arm, None)
                    self._successes[arm] = 0
                    self._failures[arm] = 0
                return
            self._failures[arm] = 0
            self._successes[arm] = 0

    def record_failure(self, arm: str) -> None:
        if arm not in self._failures:
            raise ValueError(f"unknown arm {arm!r}")
        with self._lock:
            state = self._state(arm)
            self._successes[arm] = 0
            if state is BreakerState.HALF_OPEN:
                self._opened_at[arm] = self._clock()
                return
            self._failures[arm] += 1
            if self._failures[arm] >= self._config.failure_threshold:
                self._opened_at[arm] = self._clock()
                self._failures[arm] = 0

    def snapshot(self) -> dict[str, str]:
        """Arm -> state, for ``/readyz`` and metrics."""
        with self._lock:
            return {arm: self._state(arm).value for arm in self._arms}

    def any_available(self) -> bool:
        return bool(~self.blocked_mask().all())
