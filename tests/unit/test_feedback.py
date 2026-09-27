"""Feedback ingestion under attack.

Whoever can post rewards decides what the policy learns and therefore what the traffic costs, so
every defense in :mod:`bandit_llm_router.feedback` is measured against an adversary rather than
described. The property test at the top is the core claim: a weighted median cannot be moved beyond
the next honest value, however extreme an attacker's report is -- which a mean cannot promise.
"""

from __future__ import annotations

from itertools import count

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bandit_llm_router.core.ratelimit import RateLimiter
from bandit_llm_router.feedback import (
    AggregatedReward,
    DuplicateReportError,
    ExpiredDecisionError,
    FeedbackError,
    FeedbackIngestor,
    FeedbackPolicy,
    FeedbackRateLimitedError,
    FeedbackSubmission,
    InMemoryFeedbackStore,
    SourceReport,
    TooManyReportsError,
    UnknownDecisionError,
    clip_reward,
    robust_aggregate,
    weighted_median,
)

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def make_ingestor(
    *,
    policy: FeedbackPolicy | None = None,
    limiter: RateLimiter | None = None,
    decision_at: float | None = 1_000.0,
) -> tuple[FeedbackIngestor, InMemoryFeedbackStore, FakeClock]:
    clock = FakeClock()
    store = InMemoryFeedbackStore()
    if decision_at is not None:
        store.add_decision("decision-1", decision_at)
    return (
        FeedbackIngestor(store, policy=policy, limiter=limiter, clock=clock),
        store,
        clock,
    )


_report_ids = count()


def submit(
    ingestor: FeedbackIngestor, reward: float, *, source: str, decision: str = "decision-1"
) -> AggregatedReward:
    return ingestor.submit(
        FeedbackSubmission(
            report_id=f"report-{next(_report_ids)}", decision_id=decision, reward=reward
        ),
        source=source,
    ).aggregated


# --- bounded values -----------------------------------------------------------------------------


def test_an_unbounded_report_is_clipped_and_counted() -> None:
    """One report of 1e9 would wreck a ridge posterior; clipping removes the attack outright."""
    policy = FeedbackPolicy()
    assert clip_reward(1e9, policy) == (1.0, True)
    assert clip_reward(-1e9, policy) == (0.0, True)
    assert clip_reward(0.42, policy) == (0.42, False)


def test_the_ingestor_clips_and_records_that_it_did() -> None:
    ingestor, store, _ = make_ingestor()
    result = ingestor.submit(
        FeedbackSubmission(report_id="r1", decision_id="decision-1", reward=1e6), source="rubric"
    )
    assert result.clipped is True
    assert result.row.raw_reward == 1e6
    assert result.row.reward == 1.0
    assert ingestor.clipped_total == 1
    assert store.rewards["decision-1"].reward == 1.0


def test_an_absurd_report_is_refused_by_the_request_schema() -> None:
    with pytest.raises(ValidationError):
        FeedbackSubmission(report_id="r1", decision_id="d", reward=float("inf"))
    with pytest.raises(ValidationError):
        FeedbackSubmission(report_id="r1", decision_id="d", reward=1e12)
    with pytest.raises(ValidationError):
        FeedbackSubmission(report_id="bad id!", decision_id="d", reward=0.5)


# --- robust aggregation --------------------------------------------------------------------------


@given(
    honest=st.lists(st.floats(0.0, 1.0, allow_nan=False, width=64), min_size=3, max_size=9),
    hostile=st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False, width=64),
)
def test_one_hostile_source_cannot_move_the_median_outside_the_honest_range(
    honest: list[float], hostile: float
) -> None:
    """The defining property of the defense: a mean would move linearly with ``hostile``.

    The claim is bounded, not absolute. One hostile source among at least two honest ones cannot
    push the aggregate outside the range the honest sources themselves span -- so an attacker
    reporting 1e9 buys nothing a report of ``max(honest)`` would not already have bought. It can
    still move the result *within* that range, and where the honest sources disagree wildly that
    range is wide; only corroboration narrows it.
    """
    policy = FeedbackPolicy()
    reports = [SourceReport(f"honest_{index}", value) for index, value in enumerate(honest)]
    clean = robust_aggregate(reports, policy).reward
    assert clean in honest
    for extreme in (hostile, 1e9, -1e9):
        attacked = robust_aggregate(
            [*reports, SourceReport("attacker", clip_reward(extreme, policy)[0])], policy
        ).reward
        assert min(honest) <= attacked <= max(honest)


@given(
    values=st.lists(st.floats(-100.0, 100.0, allow_nan=False, width=64), min_size=1, max_size=12),
)
def test_the_weighted_median_is_one_of_its_inputs(values: list[float]) -> None:
    weights = [1.0] * len(values)
    assert weighted_median(values, weights) in values


def test_the_weighted_median_respects_its_weights() -> None:
    assert weighted_median([0.0, 1.0], [0.9, 0.1]) == 0.0
    assert weighted_median([0.0, 1.0], [0.1, 0.9]) == 1.0
    assert weighted_median([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == 0.5


def test_the_weighted_median_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        weighted_median([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least one value"):
        weighted_median([], [])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_median([1.0], [-1.0])
    with pytest.raises(ValueError, match="must not all be zero"):
        weighted_median([1.0], [0.0])


def test_repetition_does_not_buy_an_attacker_influence() -> None:
    """Each source collapses to its own median first, so shouting louder changes nothing."""
    policy = FeedbackPolicy()
    honest = [SourceReport("rubric", 0.8), SourceReport("product", 0.75)]
    spam = [SourceReport("attacker", 0.0) for _ in range(20)]
    aggregated = robust_aggregate([*honest, *spam], policy)
    assert aggregated.reward >= 0.75
    assert aggregated.n_sources == 3
    assert aggregated.n_reports == 22


def test_a_cap_of_exactly_one_half_is_refused() -> None:
    """At half the weight one source decides the lower weighted median, so the cap is lower."""
    with pytest.raises(ValidationError):
        FeedbackPolicy(max_source_weight=0.5)
    with pytest.raises(ValidationError):
        FeedbackPolicy(max_source_weight=0.0)
    assert FeedbackPolicy().max_source_weight < 0.5


def test_a_source_weight_cap_stops_one_source_owning_the_vote() -> None:
    policy = FeedbackPolicy(max_source_weight=0.4)
    reports = [SourceReport("loud", 0.0) for _ in range(9)]
    reports.append(SourceReport("quiet", 1.0))
    # Nine of ten reports come from one source; the cap keeps it below a majority.
    assert robust_aggregate(reports, policy).reward == 1.0


def test_corroboration_buys_learning_weight() -> None:
    policy = FeedbackPolicy(single_source_weight=0.25)
    single = robust_aggregate([SourceReport("rubric", 0.6)], policy)
    assert single.weight == 0.25
    assert single.n_sources == 1
    corroborated = robust_aggregate(
        [SourceReport("rubric", 0.6), SourceReport("product", 0.62)], policy
    )
    assert corroborated.weight == 1.0
    assert corroborated.n_sources == 2


def test_aggregation_needs_at_least_one_report() -> None:
    with pytest.raises(ValueError, match="at least one report"):
        robust_aggregate([])


def test_the_aggregate_serializes_for_the_api() -> None:
    payload = AggregatedReward(reward=0.5, n_reports=3, n_sources=2, weight=1.0).as_dict()
    assert payload == {"reward": 0.5, "n_reports": 3, "n_sources": 2, "weight": 1.0}


def test_a_majority_attacker_wins_and_the_docs_say_so() -> None:
    """The honest limit of the defense: it survives a minority, not a majority."""
    policy = FeedbackPolicy()
    reports = [SourceReport("rubric", 0.9), SourceReport("product", 0.85)]
    attackers = [SourceReport(f"attacker_{index}", 0.0) for index in range(4)]
    assert robust_aggregate([*reports, *attackers], policy).reward == 0.0


# --- bounded scope -------------------------------------------------------------------------------


def test_feedback_for_an_unknown_decision_is_refused() -> None:
    ingestor, _, _ = make_ingestor(decision_at=None)
    with pytest.raises(UnknownDecisionError):
        submit(ingestor, 0.5, source="rubric")


def test_feedback_outside_the_window_is_refused() -> None:
    ingestor, _, clock = make_ingestor(policy=FeedbackPolicy(ttl_seconds=60.0))
    clock.now += 30.0
    submit(ingestor, 0.5, source="rubric")
    clock.now += 100.0
    with pytest.raises(ExpiredDecisionError):
        submit(ingestor, 0.5, source="rubric")


def test_a_replayed_report_id_is_idempotent_not_double_counted() -> None:
    ingestor, _, _ = make_ingestor()
    submission = FeedbackSubmission(report_id="r1", decision_id="decision-1", reward=0.5)
    ingestor.submit(submission, source="rubric")
    with pytest.raises(DuplicateReportError):
        ingestor.submit(submission, source="rubric")


def test_a_decision_accepts_only_so_many_reports() -> None:
    ingestor, _, _ = make_ingestor(
        policy=FeedbackPolicy(max_reports_per_decision=3, max_reports_per_source_per_decision=3)
    )
    for index in range(3):
        submit(ingestor, 0.5, source=f"source_{index}")
    with pytest.raises(TooManyReportsError, match="already has"):
        submit(ingestor, 0.5, source="source_late")


def test_one_source_cannot_outvote_the_others_by_repetition() -> None:
    ingestor, _, _ = make_ingestor(policy=FeedbackPolicy(max_reports_per_source_per_decision=2))
    submit(ingestor, 0.5, source="attacker")
    submit(ingestor, 0.4, source="attacker")
    with pytest.raises(TooManyReportsError, match="already reported"):
        submit(ingestor, 0.3, source="attacker")
    # A different source is still welcome.
    assert submit(ingestor, 0.9, source="rubric").n_sources == 2


def test_feedback_is_rate_limited_per_source() -> None:
    limiter = RateLimiter(rate=1.0, burst=2)
    ingestor, _, _ = make_ingestor(limiter=limiter)
    submit(ingestor, 0.5, source="rubric")
    submit(ingestor, 0.5, source="rubric")
    with pytest.raises(FeedbackRateLimitedError):
        submit(ingestor, 0.5, source="rubric")


def test_an_anonymous_report_is_refused() -> None:
    ingestor, _, _ = make_ingestor()
    with pytest.raises(FeedbackError, match="source is required"):
        submit(ingestor, 0.5, source="")


def test_every_refusal_is_a_feedback_error() -> None:
    for error in (
        UnknownDecisionError,
        ExpiredDecisionError,
        DuplicateReportError,
        TooManyReportsError,
        FeedbackRateLimitedError,
    ):
        assert issubclass(error, FeedbackError)
        assert issubclass(error, ValueError)


# --- the aggregate the learner sees ---------------------------------------------------------------


def test_the_stored_reward_is_re_aggregated_after_every_report() -> None:
    ingestor, store, _ = make_ingestor()
    first = submit(ingestor, 0.9, source="rubric")
    assert store.rewards["decision-1"].reward == first.reward
    second = submit(ingestor, 0.1, source="attacker")
    assert store.rewards["decision-1"].reward == second.reward
    assert second.n_sources == 2
    third = submit(ingestor, 0.85, source="product")
    # Two honest sources against one attacker: the median lands with the honest majority.
    assert third.reward >= 0.85
    assert third.n_sources == 3


def test_the_policy_is_frozen_and_versioned() -> None:
    policy = FeedbackPolicy()
    assert policy.digest_payload()["ttl_seconds"] == policy.ttl_seconds
    with pytest.raises(ValueError, match="frozen"):
        policy.ttl_seconds = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        FeedbackPolicy(max_reports_per_decision=0)
    assert ingested_policy_defaults(policy)


def ingested_policy_defaults(policy: FeedbackPolicy) -> bool:
    return (
        policy.reward_min == 0.0
        and policy.reward_max == 1.0
        and 0.0 < policy.single_source_weight <= 1.0
    )


def test_the_in_memory_store_implements_the_protocol_the_service_uses() -> None:
    store = InMemoryFeedbackStore({"d1": 5.0})
    assert store.decision_timestamp("d1") == 5.0
    assert store.decision_timestamp("missing") is None
    assert store.has_report("r") is False
    assert store.reports_for("d1") == ()
    store.set_reward("d1", AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=1.0))
    assert store.rewards["d1"].reward == 0.5
    assert ingestor_uses_the_ingestors_policy()


def ingestor_uses_the_ingestors_policy() -> bool:
    ingestor, _, _ = make_ingestor(policy=FeedbackPolicy(ttl_seconds=5.0))
    return ingestor.policy.ttl_seconds == 5.0


def test_reward_poisoning_is_visibly_worse_without_the_defenses() -> None:
    """The ablation the bench runs at scale: the same hostile stream, defenses on and off."""
    rng = np.random.default_rng(0)
    policy = FeedbackPolicy()
    naive_errors: list[float] = []
    robust_errors: list[float] = []
    for _ in range(300):
        truth = float(rng.uniform(0.2, 0.9))
        honest = [
            SourceReport(f"rubric_{index}", truth + float(rng.normal(0.0, 0.05)))
            for index in range(2)
        ]
        hostile = SourceReport("attacker", -50.0 if truth >= 0.5 else 50.0)
        naive = float(np.mean([item.reward for item in (*honest, hostile)]))
        defended = robust_aggregate(
            [
                SourceReport(item.source, clip_reward(item.reward, policy)[0])
                for item in (*honest, hostile)
            ],
            policy,
        ).reward
        naive_errors.append(abs(naive - truth))
        robust_errors.append(abs(defended - truth))
    assert float(np.mean(naive_errors)) > 10.0
    assert float(np.mean(robust_errors)) < 0.1
