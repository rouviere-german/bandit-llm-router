"""The decision log: what it holds, and the column it must never grow.

The first test is a security control, not a schema check. "We do not store prompts" is a claim that
only survives if something enforces it, so the store refuses to open against a schema with a
prompt-like column and the test proves the refusal fires. The rest checks the log is actually usable
for off-policy evaluation months later: realized propensities, full distributions, and the request
sizes needed to price arms that were never taken.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from bandit_llm_router.feedback import AggregatedReward, FeedbackRow
from bandit_llm_router.logstore import (
    _INSERT_DECISION,
    SCHEMA_VERSION,
    DecisionRecord,
    RouterLogStore,
    _is_prompt_like,
)

pytestmark = pytest.mark.unit

FEATURES = (0.5, -0.25, 1.0)
PROBABILITIES = (0.6, 0.3, 0.1)


@pytest.fixture
def store() -> Iterator[RouterLogStore]:
    with RouterLogStore() as opened:
        yield opened


def record(decision_id: str = "d1", **overrides: object) -> DecisionRecord:
    values: dict[str, object] = {
        "decision_id": decision_id,
        "logged_at": 1_700_000_000.0,
        "tenant_id": "tenant_a",
        "policy_version": "linucb-abc123",
        "policy_step": 7,
        "feature_version": "ctx-v1-deadbeef",
        "context_digest": "a" * 32,
        "arm": "haiku",
        "arm_index": 0,
        "propensity": 0.6,
        "probabilities": PROBABILITIES,
        "features": FEATURES,
        "input_tokens": 120,
        "max_output_tokens": 512,
        "estimated_cost_micros": 1_600,
        "ceiling_cost_micros": 2_680,
    }
    values.update(overrides)
    return DecisionRecord.model_validate(values)


# --- the no-prompt guarantee ---------------------------------------------------------------------


def test_no_table_has_a_column_that_could_hold_a_prompt(store: RouterLogStore) -> None:
    for table in ("decisions", "feedback", "rewards", "store_meta"):
        columns = store.columns(table)
        assert columns
        for column in columns:
            assert not _is_prompt_like(column)


@pytest.mark.parametrize(
    ("column", "offends"),
    [
        ("prompt", True),
        ("prompt_text", True),
        ("userprompt", True),
        ("request_body", True),
        ("response_content", True),
        ("assistant_message", True),
        ("raw_payload", True),
        # The keyed digest is the one prompt-derived value that is allowed, and its name contains
        # "text" by accident; a substring check would reject the whole schema.
        ("context_digest", False),
        ("max_output_tokens", False),
        ("input_tokens", False),
        ("propensity", False),
    ],
)
def test_the_guard_matches_name_tokens_not_substrings(column: str, offends: bool) -> None:
    assert _is_prompt_like(column) is offends


def test_the_store_refuses_to_open_against_a_prompt_column(tmp_path: Path) -> None:
    """The guard is what turns "we do not store prompts" into something enforced."""
    import duckdb

    path = tmp_path / "tainted.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE audit_notes (prompt_text VARCHAR)")
    connection.close()
    with pytest.raises(RuntimeError, match="must not contain prompt-like columns"):
        RouterLogStore(path)


def test_the_record_itself_has_no_field_for_text() -> None:
    fields = set(DecisionRecord.model_fields)
    assert not (fields & {"prompt", "text", "content", "message", "body"})
    with pytest.raises(ValidationError):
        record(prompt="secret")


def test_the_context_digest_is_the_only_prompt_derived_value() -> None:
    row = record()
    assert row.context_digest == "a" * 32
    with pytest.raises(ValidationError):
        record(context_digest="not hex at all")
    with pytest.raises(ValidationError):
        record(context_digest="ab")  # too short to be a digest


# --- the contract on a row -----------------------------------------------------------------------


def test_a_row_that_cannot_be_reweighted_is_refused() -> None:
    with pytest.raises(ValidationError):
        record(propensity=0.0)
    with pytest.raises(ValidationError):
        record(propensity=1.5)
    with pytest.raises(ValidationError):
        record(probabilities=(1.0,))  # a single arm is not a decision
    with pytest.raises(ValidationError):
        record(arm_index=-1)
    with pytest.raises(ValidationError):
        record(input_tokens=0)


def test_a_row_is_frozen() -> None:
    row = record()
    with pytest.raises(ValueError, match="frozen"):
        row.propensity = 0.1  # type: ignore[misc]


def test_the_store_stamps_its_schema_version(store: RouterLogStore) -> None:
    assert SCHEMA_VERSION == 1
    assert store.path == ":memory:"


# --- writes and reads ----------------------------------------------------------------------------


def test_a_decision_round_trips_exactly(store: RouterLogStore) -> None:
    store.record_decision(record())
    loaded = store.decision("d1")
    assert loaded is not None
    assert loaded.features == FEATURES
    assert loaded.probabilities == PROBABILITIES
    assert loaded.propensity == 0.6
    assert loaded.policy_step == 7
    assert loaded.actual_cost_micros is None
    assert store.decision("missing") is None
    assert store.count_decisions() == 1


def test_decisions_can_be_written_in_batches(store: RouterLogStore) -> None:
    store.record_decisions([record(f"d{index}") for index in range(5)])
    assert store.count_decisions() == 5
    store.record_decisions([])
    assert store.count_decisions() == 5


def test_the_actual_cost_is_settled_after_the_call(store: RouterLogStore) -> None:
    store.record_decision(record())
    store.set_actual_cost("d1", 1_450)
    loaded = store.decision("d1")
    assert loaded is not None
    assert loaded.actual_cost_micros == 1_450
    with pytest.raises(ValueError, match="non-negative"):
        store.set_actual_cost("d1", -1)


def test_arm_counts_and_spend_are_grouped_for_the_dashboards(store: RouterLogStore) -> None:
    store.record_decisions(
        [
            record("d1", arm="haiku", arm_index=0, estimated_cost_micros=1_000),
            record("d2", arm="haiku", arm_index=0, estimated_cost_micros=1_000),
            record(
                "d3",
                arm="opus",
                arm_index=2,
                propensity=0.1,
                estimated_cost_micros=9_000,
                actual_cost_micros=8_500,
                tenant_id="tenant_b",
            ),
        ]
    )
    assert store.arm_counts() == {"haiku": 2, "opus": 1}
    # Actual cost wins where it is known; the estimate stands in where it is not.
    assert store.spend_by_tenant() == {"tenant_a": 2_000, "tenant_b": 8_500}


def test_the_feature_matrix_is_available_for_drift_monitoring(store: RouterLogStore) -> None:
    assert store.feature_matrix().shape == (0, 0)
    store.record_decisions(
        [
            record("d1", logged_at=1.0, features=(0.1, 0.2, 1.0)),
            record("d2", logged_at=2.0, features=(0.3, 0.4, 1.0)),
        ]
    )
    newest = store.feature_matrix(limit=1)
    assert newest.shape == (1, 3)
    assert newest[0] == pytest.approx([0.3, 0.4, 1.0])
    oldest = store.feature_matrix(limit=1, newest_first=False)
    assert oldest[0] == pytest.approx([0.1, 0.2, 1.0])
    with pytest.raises(ValueError, match="limit must be"):
        store.feature_matrix(limit=0)


# --- the feedback store protocol -----------------------------------------------------------------


def test_the_store_answers_the_questions_the_ingestor_asks(store: RouterLogStore) -> None:
    store.record_decision(record())
    assert store.decision_timestamp("d1") == 1_700_000_000.0
    assert store.decision_timestamp("nope") is None
    assert store.has_report("r1") is False
    assert store.reports_for("d1") == ()
    row = FeedbackRow(
        report_id="r1",
        decision_id="d1",
        source="rubric",
        raw_reward=1e6,
        reward=1.0,
        received_at=time.time(),
    )
    store.record_feedback(row)
    assert store.has_report("r1") is True
    stored = store.reports_for("d1")
    assert len(stored) == 1
    # Both the claim and what was used are kept, so an audit can see the difference.
    assert stored[0].raw_reward == 1e6
    assert stored[0].reward == 1.0


def test_the_aggregated_reward_is_replaced_not_appended(store: RouterLogStore) -> None:
    store.record_decision(record())
    store.set_reward("d1", AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=0.5))
    store.set_reward("d1", AggregatedReward(reward=0.7, n_reports=2, n_sources=2, weight=1.0))
    assert store.count_rewards() == 1
    rows = store.fetch_logged_rows(n_arms=3)
    assert rows.batch.rewards == pytest.approx([0.7])


# --- the OPE batch -------------------------------------------------------------------------------


def test_only_rewarded_decisions_reach_the_evaluation_batch(store: RouterLogStore) -> None:
    """An importance-weighted estimator has nothing to weight without an outcome."""
    store.record_decisions([record("d1"), record("d2"), record("d3")])
    store.set_reward("d1", AggregatedReward(reward=0.8, n_reports=1, n_sources=1, weight=1.0))
    store.set_reward("d3", AggregatedReward(reward=0.2, n_reports=1, n_sources=1, weight=1.0))
    rows = store.fetch_logged_rows(n_arms=3)
    assert rows.decision_ids == ("d1", "d3")
    assert rows.batch.n == 2
    assert rows.batch.rewards == pytest.approx([0.8, 0.2])


def test_the_batch_carries_the_sizes_needed_to_price_untaken_arms(
    store: RouterLogStore,
) -> None:
    store.record_decision(record("d1", input_tokens=250, max_output_tokens=900))
    store.set_reward("d1", AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=1.0))
    rows = store.fetch_logged_rows(n_arms=3)
    assert list(rows.input_tokens) == [250]
    assert list(rows.max_output_tokens) == [900]
    assert rows.batch.features.shape == (1, 3)
    assert rows.batch.propensities == pytest.approx([0.6])


def test_the_batch_can_be_restricted_to_named_policy_versions(store: RouterLogStore) -> None:
    store.record_decisions(
        [
            record("d1", policy_version="linucb-old"),
            record("d2", policy_version="linucb-new"),
        ]
    )
    for decision_id in ("d1", "d2"):
        store.set_reward(
            decision_id, AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=1.0)
        )
    assert store.fetch_logged_rows(n_arms=3, policy_versions=["linucb-new"]).decision_ids == ("d2",)
    assert store.logged_batch(n_arms=3).n == 2
    assert store.fetch_logged_rows(n_arms=3, limit=1).batch.n == 1
    with pytest.raises(ValueError, match="limit must be"):
        store.fetch_logged_rows(n_arms=3, limit=0)


def test_the_batch_can_be_restricted_to_a_window_of_policy_steps(store: RouterLogStore) -> None:
    """Screening a live learner's own log needs a way to ask for a stretch of it, not all of it.

    ``policy_version`` cannot do this job: it hashes configuration only, so every row a learning
    LinUCB ever wrote carries the same one however far its posteriors have moved. ``policy_step``
    counts the updates, so a step window is the only handle on "the log while the policy was nearly
    still".
    """
    reward = AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=1.0)
    for step in range(10):
        store.record_decision(
            record(f"d{step}", policy_step=step, logged_at=1_700_000_000.0 + step)
        )
        store.set_reward(f"d{step}", reward)
    windowed = store.fetch_logged_rows(n_arms=3, min_policy_step=3, max_policy_step=5)
    assert windowed.decision_ids == ("d3", "d4", "d5")
    assert list(windowed.policy_steps) == [3, 4, 5]
    assert store.fetch_logged_rows(n_arms=3, min_policy_step=8).decision_ids == ("d8", "d9")
    assert store.fetch_logged_rows(n_arms=3, max_policy_step=1).decision_ids == ("d0", "d1")
    assert list(store.fetch_logged_rows(n_arms=3).policy_steps) == list(range(10))


def test_a_policy_step_window_that_cannot_contain_a_row_is_refused(store: RouterLogStore) -> None:
    store.record_decision(record("d1", policy_step=4))
    store.set_reward("d1", AggregatedReward(reward=0.5, n_reports=1, n_sources=1, weight=1.0))
    with pytest.raises(ValueError, match="min_policy_step"):
        store.fetch_logged_rows(n_arms=3, min_policy_step=-1)
    with pytest.raises(ValueError, match="max_policy_step"):
        store.fetch_logged_rows(n_arms=3, min_policy_step=5, max_policy_step=4)
    with pytest.raises(ValueError, match="no rewarded decisions"):
        store.fetch_logged_rows(n_arms=3, min_policy_step=9)


def test_an_empty_log_refuses_to_pretend_it_can_evaluate_anything(
    store: RouterLogStore,
) -> None:
    with pytest.raises(ValueError, match="no rewarded decisions"):
        store.fetch_logged_rows(n_arms=3)


def test_the_insert_statement_has_one_placeholder_per_column(store: RouterLogStore) -> None:
    """The INSERT is a literal, so something has to keep its arity honest.

    Writing the placeholders out by hand is what lets the statement be a plain string constant
    instead of dynamic SQL a scanner has to be told to ignore. The cost is that adding a column to
    ``DecisionRecord`` could leave the statement one short, and DuckDB's error for that is not
    obvious. This test is that cost paid.
    """
    assert _INSERT_DECISION.count("?") == len(record().row())
    assert _INSERT_DECISION.count("?") == len(store.columns("decisions"))
    # And it really is the statement the store uses: a row written through it reads back whole.
    store.record_decision(record("round-trip"))
    assert store.decision("round-trip") == record("round-trip")


def test_the_store_survives_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "decisions.duckdb"
    with RouterLogStore(path) as first:
        first.record_decision(record())
    with RouterLogStore(path) as second:
        assert second.count_decisions() == 1
        assert np.asarray(second.feature_matrix()).shape == (1, 3)
