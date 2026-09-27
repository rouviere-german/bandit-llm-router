"""Data contract for the decision log, with PII-safe failure reporting.

The log is the asset in this repo: every later claim about a policy's value is computed from it. A
row that violates its contract -- a propensity of zero, a reward outside the reward range, a
duplicate decision id -- is not a cosmetic problem, it silently corrupts an estimate months later.
So the contract is checked at the boundary, and it is deliberately strict:

* ``strict = True`` means an *extra* column fails validation. That is the mechanism that keeps
  prompts out: a well-meaning change that starts attaching request text to the log cannot pass.
* ``propensity`` must be in (0, 1]. A zero-propensity row cannot be reweighted and must never be
  written.
* ``reward`` is nullable (feedback is delayed) but bounded when present.
* ``decision_id`` must be unique, so a replayed write cannot double-count an outcome.

Failures are summarized as (column, check, count). The offending values are never copied into the
exception, because in a system whose whole point is not to store prompts, an error message is a
terrible place to start.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing.pandas import Series

if TYPE_CHECKING:
    from bandit_llm_router.logstore import DecisionRecord

__all__ = [
    "DataContractError",
    "DecisionLogSchema",
    "decision_log_frame",
    "validate_decision_log",
]


class DataContractError(ValueError):
    """A decision-log frame violated its contract. ``failures`` holds (column, check, count)."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"decision log contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class DecisionLogSchema(pa.DataFrameModel):
    """Contract for a batch of logged routing decisions joined with their rewards."""

    decision_id: Series[str] = pa.Field(unique=True, str_matches=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    logged_at: Series[float] = pa.Field(gt=0.0)
    tenant_id: Series[str] = pa.Field(str_matches=r"^[a-z0-9][a-z0-9_-]*$")
    policy_version: Series[str] = pa.Field(str_length={"min_value": 1, "max_value": 128})
    feature_version: Series[str] = pa.Field(str_length={"min_value": 1, "max_value": 128})
    arm: Series[str] = pa.Field(str_matches=r"^[a-z][a-z0-9_]*$")
    arm_index: Series[int] = pa.Field(ge=0, le=63)
    propensity: Series[float] = pa.Field(gt=0.0, le=1.0)
    estimated_cost_micros: Series[int] = pa.Field(ge=0)
    ceiling_cost_micros: Series[int] = pa.Field(ge=0)
    reward: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)

    class Config:
        strict = True
        name = "decision_log"

    @pa.dataframe_check
    @classmethod
    def ceiling_is_not_below_the_estimate(cls, frame: pd.DataFrame) -> Series[bool]:
        """A reservation must bound the cost it settles, or the budget cap is not a bound."""
        return Series[bool](frame["ceiling_cost_micros"] >= frame["estimated_cost_micros"])


def validate_decision_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily (report every failure at once) and raise a summarized error."""
    try:
        return DecisionLogSchema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].astype(str)
        counts = cases.value_counts().sort_index()
        failures: list[dict[str, str | int]] = [
            {"column": column, "check": check, "count": int(count)}
            for (column, check), count in counts.items()
        ]
        raise DataContractError(failures) from None


def decision_log_frame(
    records: Sequence[DecisionRecord], rewards: Sequence[float | None] | None = None
) -> pd.DataFrame:
    """Build the contract's frame from decision records and their (possibly missing) rewards.

    Only contract columns are included: feature vectors and probability vectors stay out of the
    frame, and nothing prompt-derived is here to begin with.
    """
    if rewards is not None and len(rewards) != len(records):
        raise ValueError("rewards must align with records")
    reward_values = list(rewards) if rewards is not None else [None] * len(records)
    return pd.DataFrame(
        {
            "decision_id": pd.Series([r.decision_id for r in records], dtype="str"),
            "logged_at": pd.Series([r.logged_at for r in records], dtype="float64"),
            "tenant_id": pd.Series([r.tenant_id for r in records], dtype="str"),
            "policy_version": pd.Series([r.policy_version for r in records], dtype="str"),
            "feature_version": pd.Series([r.feature_version for r in records], dtype="str"),
            "arm": pd.Series([r.arm for r in records], dtype="str"),
            "arm_index": pd.Series([r.arm_index for r in records], dtype="int64"),
            "propensity": pd.Series([r.propensity for r in records], dtype="float64"),
            "estimated_cost_micros": pd.Series(
                [r.estimated_cost_micros for r in records], dtype="int64"
            ),
            "ceiling_cost_micros": pd.Series(
                [r.ceiling_cost_micros for r in records], dtype="int64"
            ),
            "reward": pd.Series(reward_values, dtype="float64"),
        }
    )
