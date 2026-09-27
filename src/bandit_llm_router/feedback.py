"""Delayed feedback ingestion that assumes some of the feedback is hostile.

Rewards arrive after the fact, over an authenticated endpoint, from sources the router does not
control -- an automated rubric, a product signal, a human review queue. Feedback is therefore the
most attackable surface in the system: whoever can post rewards can steer which model gets traffic
and how much it costs. Five controls stack up here, and each is measured rather than asserted:

1. **Bounded values.** Every reported reward is clipped into ``[reward_min, reward_max]``. An
   unbounded report is the cheapest possible attack (one report of ``1e9`` wrecks a ridge
   posterior), and clipping removes it outright. Clipped reports are counted, not silently accepted.
2. **Bounded volume.** The caller is rate limited per API key, and a decision accepts at most
   ``max_reports_per_decision`` reports, at most ``max_reports_per_source_per_decision`` from any
   one source. An attacker cannot outvote honest sources by repetition.
3. **Bounded scope.** A report is accepted only for a decision that exists and is inside the
   feedback TTL, and a ``report_id`` is accepted once, so retries are idempotent and a replayed
   batch cannot double-count.
4. **Robust aggregation.** Each source is reduced to its own median first, then sources are combined
   with a *weighted median* whose weights are capped at ``max_source_weight``, which must be
   strictly below one half. A cap of exactly one half would be no defense at all: the lower weighted
   median is decided by whoever first reaches half the total weight, so a source sitting on the cap
   would own the result no matter how many honest sources disagreed. Below one half, one dishonest
   source among at least two honest ones cannot move the outcome however extreme its values, while a
   mean would move linearly with them. One honest source against one hostile one is a tie the
   aggregator cannot break; that is a minimum-source requirement (docs/THREAT_MODEL.md), not
   something clipping fixes.
5. **Trust-weighted learning.** A reward backed by a single source updates the posterior with weight
   ``single_source_weight`` instead of 1. Corroboration buys influence.

What this does *not* do: it cannot recover the truth when the attacker controls a majority of
sources, and it cannot detect a sophisticated attacker who reports plausible-but-wrong values. Both
limits are in docs/THREAT_MODEL.md, and ``bench/bench_router.py`` measures what the defenses are
worth by running the same poisoned stream with them on and off.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bandit_llm_router.core.ratelimit import RateLimiter

__all__ = [
    "AggregatedReward",
    "DuplicateReportError",
    "ExpiredDecisionError",
    "FeedbackError",
    "FeedbackIngestor",
    "FeedbackPolicy",
    "FeedbackRateLimitedError",
    "FeedbackRow",
    "FeedbackStore",
    "FeedbackSubmission",
    "InMemoryFeedbackStore",
    "IngestResult",
    "SourceReport",
    "TooManyReportsError",
    "UnknownDecisionError",
    "clip_reward",
    "robust_aggregate",
    "weighted_median",
]

FloatArray = npt.NDArray[np.float64]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]


class FeedbackError(ValueError):
    """A feedback report was refused."""


class UnknownDecisionError(FeedbackError):
    """No decision with that id was logged (or it was logged by another deployment)."""


class ExpiredDecisionError(FeedbackError):
    """The decision is older than the feedback TTL."""


class DuplicateReportError(FeedbackError):
    """That report id was already ingested; the ingest is idempotent."""


class TooManyReportsError(FeedbackError):
    """The decision, or this source on this decision, has reached its report limit."""


class FeedbackRateLimitedError(FeedbackError):
    """The source exceeded its feedback rate limit."""


class FeedbackPolicy(BaseModel):
    """Ingestion limits and the aggregation rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reward_min: float = 0.0
    reward_max: float = 1.0
    ttl_seconds: float = Field(default=86_400.0, gt=0.0)
    max_reports_per_decision: int = Field(default=8, ge=1)
    max_reports_per_source_per_decision: int = Field(default=2, ge=1)
    # Strictly below one half: see the module docstring. At exactly one half a single source decides
    # the lower weighted median, which is the attack this cap exists to prevent.
    max_source_weight: float = Field(default=0.4, gt=0.0, lt=0.5)
    single_source_weight: float = Field(default=0.5, gt=0.0, le=1.0)

    def digest_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class FeedbackSubmission(BaseModel):
    """What a caller posts. ``reward`` is accepted loosely and then clipped, never trusted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: Identifier
    decision_id: Identifier
    reward: float = Field(ge=-1e9, le=1e9, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class FeedbackRow:
    """One stored report."""

    report_id: str
    decision_id: str
    source: str
    raw_reward: float
    reward: float
    received_at: float


@dataclass(frozen=True, slots=True)
class SourceReport:
    """A clipped report as the aggregator sees it."""

    source: str
    reward: float


@dataclass(frozen=True, slots=True)
class AggregatedReward:
    """The reward the learner will use, plus how much corroboration stands behind it."""

    reward: float
    n_reports: int
    n_sources: int
    weight: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "reward": self.reward,
            "n_reports": self.n_reports,
            "n_sources": self.n_sources,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one accepted submission."""

    row: FeedbackRow
    clipped: bool
    aggregated: AggregatedReward


def clip_reward(value: float, policy: FeedbackPolicy) -> tuple[float, bool]:
    """Clip a reported reward into the allowed range; also report whether it was out of range."""
    clipped = min(policy.reward_max, max(policy.reward_min, value))
    return clipped, clipped != value


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Lower weighted median: the smallest value whose cumulative weight reaches half the total.

    Unlike a weighted mean, moving one input to +/- infinity cannot move the output beyond the next
    value in order, which is exactly the property the poisoning defense needs.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        raise ValueError("at least one value is required")
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(weight_array.sum())
    if total <= 0.0:
        raise ValueError("weights must not all be zero")
    order = np.argsort(value_array, kind="stable")
    cumulative = np.cumsum(weight_array[order])
    index = int(np.searchsorted(cumulative, total / 2.0, side="left"))
    return float(value_array[order[min(index, value_array.size - 1)]])


def robust_aggregate(
    reports: Sequence[SourceReport], policy: FeedbackPolicy | None = None
) -> AggregatedReward:
    """Combine reports: per-source median, then a weight-capped weighted median across sources."""
    settings = policy if policy is not None else FeedbackPolicy()
    if not reports:
        raise ValueError("at least one report is required")
    by_source: dict[str, list[float]] = {}
    for report in reports:
        by_source.setdefault(report.source, []).append(report.reward)
    sources = sorted(by_source)
    summaries = [float(np.median(by_source[source])) for source in sources]
    counts = np.asarray([len(by_source[source]) for source in sources], dtype=np.float64)
    weights = counts / counts.sum()
    cap = settings.max_source_weight
    if len(sources) > 1 and float(weights.max()) > cap:
        # Cap the loudest sources and give the freed mass to the others, proportionally.
        capped = np.minimum(weights, cap)
        deficit = 1.0 - float(capped.sum())
        headroom = cap - capped
        if float(headroom.sum()) > 0.0:
            capped = capped + deficit * headroom / float(headroom.sum())
        weights = capped / float(capped.sum())
    reward = weighted_median(summaries, [float(value) for value in weights])
    n_sources = len(sources)
    weight = 1.0 if n_sources > 1 else settings.single_source_weight
    return AggregatedReward(
        reward=reward,
        n_reports=len(reports),
        n_sources=n_sources,
        weight=weight,
    )


class FeedbackStore(Protocol):
    """Storage the ingestor needs. Implemented in memory here and on DuckDB in ``logstore``."""

    def decision_timestamp(self, decision_id: str) -> float | None: ...

    def has_report(self, report_id: str) -> bool: ...

    def reports_for(self, decision_id: str) -> tuple[FeedbackRow, ...]: ...

    def record_feedback(self, row: FeedbackRow) -> None: ...

    def set_reward(self, decision_id: str, aggregated: AggregatedReward) -> None: ...


class InMemoryFeedbackStore:
    """Minimal :class:`FeedbackStore` for tests, simulations and the offline demo."""

    def __init__(self, decisions: Mapping[str, float] | None = None) -> None:
        self._decisions: dict[str, float] = dict(decisions or {})
        self._reports: dict[str, list[FeedbackRow]] = {}
        self._report_ids: set[str] = set()
        self.rewards: dict[str, AggregatedReward] = {}

    def add_decision(self, decision_id: str, timestamp: float) -> None:
        self._decisions[decision_id] = timestamp

    def decision_timestamp(self, decision_id: str) -> float | None:
        return self._decisions.get(decision_id)

    def has_report(self, report_id: str) -> bool:
        return report_id in self._report_ids

    def reports_for(self, decision_id: str) -> tuple[FeedbackRow, ...]:
        return tuple(self._reports.get(decision_id, ()))

    def record_feedback(self, row: FeedbackRow) -> None:
        self._reports.setdefault(row.decision_id, []).append(row)
        self._report_ids.add(row.report_id)

    def set_reward(self, decision_id: str, aggregated: AggregatedReward) -> None:
        self.rewards[decision_id] = aggregated


class FeedbackIngestor:
    """Validates, rate limits, clips, stores and re-aggregates feedback for one decision."""

    def __init__(
        self,
        store: FeedbackStore,
        *,
        policy: FeedbackPolicy | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._policy = policy if policy is not None else FeedbackPolicy()
        self._limiter = limiter
        self._clock = clock
        self.clipped_total = 0

    @property
    def policy(self) -> FeedbackPolicy:
        return self._policy

    def submit(self, submission: FeedbackSubmission, *, source: str) -> IngestResult:
        """Ingest one report. Raises a :class:`FeedbackError` subclass on every refusal."""
        if not source:
            raise FeedbackError("a feedback source is required")
        if self._limiter is not None and not self._limiter.check(f"feedback:{source}").allowed:
            raise FeedbackRateLimitedError(f"source {source} exceeded its feedback rate limit")
        if self._store.has_report(submission.report_id):
            raise DuplicateReportError(f"report {submission.report_id} was already ingested")
        logged_at = self._store.decision_timestamp(submission.decision_id)
        if logged_at is None:
            raise UnknownDecisionError(f"decision {submission.decision_id} is not in the log")
        now = self._clock()
        if now - logged_at > self._policy.ttl_seconds:
            raise ExpiredDecisionError(
                f"decision {submission.decision_id} is outside the feedback window"
            )
        existing = self._store.reports_for(submission.decision_id)
        if len(existing) >= self._policy.max_reports_per_decision:
            raise TooManyReportsError(
                f"decision {submission.decision_id} already has "
                f"{len(existing)} reports (limit {self._policy.max_reports_per_decision})"
            )
        from_source = sum(1 for row in existing if row.source == source)
        if from_source >= self._policy.max_reports_per_source_per_decision:
            raise TooManyReportsError(
                f"source {source} already reported {from_source} times on this decision"
            )
        reward, was_clipped = clip_reward(submission.reward, self._policy)
        if was_clipped:
            self.clipped_total += 1
        row = FeedbackRow(
            report_id=submission.report_id,
            decision_id=submission.decision_id,
            source=source,
            raw_reward=submission.reward,
            reward=reward,
            received_at=now,
        )
        self._store.record_feedback(row)
        aggregated = robust_aggregate(
            [SourceReport(item.source, item.reward) for item in (*existing, row)], self._policy
        )
        self._store.set_reward(submission.decision_id, aggregated)
        return IngestResult(row=row, clipped=was_clipped, aggregated=aggregated)
