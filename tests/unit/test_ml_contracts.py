"""The decision-log data contract.

The log is the asset in this repo: every claim about a policy's value is computed from it months
after the fact. A row that violates its contract does not look broken, it quietly corrupts an
estimate, so the contract is checked at the boundary and is deliberately strict. Two of its rules
are
security controls rather than hygiene:

* ``strict = True`` fails on an *extra* column, which is the mechanism that keeps prompts out. A
  well-meaning change that starts attaching request text to the log cannot pass validation.
* failures are reported as (column, check, count) and never carry the offending values, because in a
  system whose whole point is not to store prompts, an error message is a terrible place to start.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bandit_llm_router.logstore import DecisionRecord
from bandit_llm_router.ml.contracts import (
    DataContractError,
    DecisionLogSchema,
    decision_log_frame,
    validate_decision_log,
)

pytestmark = pytest.mark.unit


def record(decision_id: str = "d1", **overrides: object) -> DecisionRecord:
    values: dict[str, object] = {
        "decision_id": decision_id,
        "logged_at": 1_700_000_000.0,
        "tenant_id": "tenant_a",
        "policy_version": "linucb-abc123",
        "policy_step": 3,
        "feature_version": "ctx-v1-deadbeef",
        "context_digest": "a" * 32,
        "arm": "haiku",
        "arm_index": 0,
        "propensity": 0.6,
        "probabilities": (0.6, 0.3, 0.1),
        "features": (0.5, -0.25, 1.0),
        "input_tokens": 120,
        "max_output_tokens": 512,
        "estimated_cost_micros": 1_600,
        "ceiling_cost_micros": 2_680,
    }
    values.update(overrides)
    return DecisionRecord.model_validate(values)


def frame(count: int = 3, rewards: list[float | None] | None = None) -> pd.DataFrame:
    records = [record(f"d{index}") for index in range(count)]
    return decision_log_frame(records, rewards)


# --- the happy path -------------------------------------------------------------------------------


def test_a_well_formed_log_validates() -> None:
    validated = validate_decision_log(frame(3, [0.8, 0.5, None]))
    assert len(validated) == 3
    assert validated["reward"].isna().sum() == 1


def test_a_delayed_reward_is_allowed_to_be_missing() -> None:
    """Feedback arrives after the fact, so a row without an outcome is normal, not invalid."""
    validated = validate_decision_log(frame(2, [None, None]))
    assert validated["reward"].isna().all()


def test_the_frame_holds_only_contract_columns() -> None:
    columns = set(frame().columns)
    assert columns == set(DecisionLogSchema.to_schema().columns)
    # Feature vectors and probability vectors stay out of the frame entirely.
    assert "features" not in columns
    assert "probabilities" not in columns


# --- what the contract refuses --------------------------------------------------------------------


def test_an_extra_column_fails_which_is_what_keeps_prompts_out() -> None:
    tainted = frame()
    tainted["prompt"] = ["secret one", "secret two", "secret three"]
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(tainted)
    # A column the schema does not know about is a schema-level failure, so it is reported against
    # the schema rather than against a column name.
    assert [item["check"] for item in failure.value.failures] == ["column_in_schema"]
    # The message counts failures; it never repeats what was in the column.
    assert "secret" not in str(failure.value)
    assert "secret" not in str(failure.value.failures)


def test_a_zero_propensity_row_is_refused() -> None:
    """It cannot be reweighted, so it must never reach the log in the first place."""
    broken = frame()
    broken.loc[0, "propensity"] = 0.0
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(broken)
    assert any(item["column"] == "propensity" for item in failure.value.failures)


def test_a_propensity_above_one_is_refused() -> None:
    broken = frame()
    broken.loc[1, "propensity"] = 1.5
    with pytest.raises(DataContractError):
        validate_decision_log(broken)


def test_a_reward_outside_the_reward_range_is_refused() -> None:
    broken = frame(2, [0.5, 1.5])
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(broken)
    assert any(item["column"] == "reward" for item in failure.value.failures)


def test_a_duplicate_decision_id_is_refused() -> None:
    """A replayed write must not be able to double-count an outcome."""
    duplicated = pd.concat([frame(1, [0.5]), frame(1, [0.5])], ignore_index=True)
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(duplicated)
    assert any(item["column"] == "decision_id" for item in failure.value.failures)


def test_a_ceiling_below_the_estimate_is_refused() -> None:
    """A reservation must bound the cost it settles, or the budget cap is not a bound."""
    broken = frame()
    broken.loc[0, "ceiling_cost_micros"] = 1
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(broken)
    assert any(
        "ceiling_is_not_below_the_estimate" in str(item["check"]) for item in failure.value.failures
    )


def test_a_malformed_identifier_is_refused() -> None:
    broken = frame()
    broken.loc[0, "tenant_id"] = "Tenant With Spaces"
    with pytest.raises(DataContractError):
        validate_decision_log(broken)


def test_every_failure_is_reported_at_once() -> None:
    """Lazy validation, so one broken column does not hide the next."""
    broken = frame(3, [0.5, 0.5, 0.5])
    broken.loc[0, "propensity"] = 0.0
    broken.loc[1, "arm_index"] = -1
    broken.loc[2, "tenant_id"] = "BAD"
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(broken)
    columns = {str(item["column"]) for item in failure.value.failures}
    assert {"propensity", "arm_index", "tenant_id"} <= columns
    assert "3 failure(s)" in str(failure.value)


def test_the_error_counts_failures_per_check() -> None:
    broken = frame(3, [0.5, 0.5, 0.5])
    broken["propensity"] = 0.0
    with pytest.raises(DataContractError) as failure:
        validate_decision_log(broken)
    (item,) = [row for row in failure.value.failures if row["column"] == "propensity"]
    assert int(item["count"]) == 3


# --- building the frame ---------------------------------------------------------------------------


def test_rewards_must_align_with_the_records() -> None:
    with pytest.raises(ValueError, match="must align"):
        decision_log_frame([record()], [0.5, 0.5])


def test_an_empty_log_is_an_empty_frame() -> None:
    built = decision_log_frame([])
    assert built.empty
    assert set(built.columns) == set(DecisionLogSchema.to_schema().columns)


def test_the_frame_types_are_the_ones_the_schema_expects() -> None:
    built = frame(2, [0.5, None])
    assert built["arm_index"].dtype == "int64"
    assert built["propensity"].dtype == "float64"
    assert built["reward"].dtype == "float64"
    assert pd.api.types.is_string_dtype(built["decision_id"])
    assert pd.api.types.is_string_dtype(built["arm"])
