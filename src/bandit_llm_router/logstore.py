"""The decision log: DuckDB tables holding what off-policy evaluation needs, and nothing else.

Three tables:

* ``decisions`` -- one row per routing decision: the hashed context digest, the feature vector, the
  arm, **the propensity actually realized**, the full action distribution, the policy version and
  step, the estimated and worst-case cost, and the reproducibility fields (feature version, workload
  digest, simulator seed).
* ``feedback`` -- one row per accepted report, with both the raw and the clipped reward, so an audit
  can see what a source claimed as well as what was used.
* ``rewards`` -- the aggregated reward per decision, with how many reports and sources produced it
  and the learning weight it earned.

**There is no prompt column, and no column that could hold one.** Prompts are read once to build
features and then dropped; what identifies a context in the log is the keyed digest from the
featurizer. ``tests/unit/test_logstore.py`` asserts the schema stays that way, because "we do not
store prompts" is a claim that only survives if something enforces it.

The store also implements the :class:`~bandit_llm_router.feedback.FeedbackStore` protocol, so the
ingestor works against DuckDB in the service and against an in-memory dict in tests and simulations
without changing the ingestion logic being tested.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bandit_llm_router.feedback import AggregatedReward, FeedbackRow
from bandit_llm_router.ope.estimators import LoggedBatch

__all__ = ["SCHEMA_VERSION", "DecisionRecord", "LoggedRows", "RouterLogStore"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SCHEMA_VERSION: Final = 1
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]
# Column names are checked against these at open time: a prompt must never reach the log. The match
# is on whole name *tokens* (``prompt_text`` offends, ``context_digest`` does not), because a
# substring match would reject the one prompt-derived value that is deliberately allowed -- the
# keyed digest in ``context_digest``, whose name happens to contain "text".
_FORBIDDEN_COLUMN_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "prompts",
        "text",
        "content",
        "contents",
        "message",
        "messages",
        "body",
        "payload",
        "completion",
        "response",
    }
)
_COLUMN_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")

_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id VARCHAR PRIMARY KEY,
    logged_at DOUBLE NOT NULL,
    tenant_id VARCHAR NOT NULL,
    policy_version VARCHAR NOT NULL,
    policy_step BIGINT NOT NULL,
    feature_version VARCHAR NOT NULL,
    context_digest VARCHAR NOT NULL,
    arm VARCHAR NOT NULL,
    arm_index INTEGER NOT NULL,
    propensity DOUBLE NOT NULL,
    probabilities DOUBLE[] NOT NULL,
    features DOUBLE[] NOT NULL,
    input_tokens BIGINT NOT NULL,
    max_output_tokens BIGINT NOT NULL,
    estimated_cost_micros BIGINT NOT NULL,
    ceiling_cost_micros BIGINT NOT NULL,
    actual_cost_micros BIGINT,
    workload_digest VARCHAR,
    simulator_seed BIGINT,
    breaker_adjusted BOOLEAN NOT NULL,
    served_by_fallback BOOLEAN NOT NULL
)
"""
_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    report_id VARCHAR PRIMARY KEY,
    decision_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    raw_reward DOUBLE NOT NULL,
    reward DOUBLE NOT NULL,
    received_at DOUBLE NOT NULL
)
"""
_REWARDS_DDL = """
CREATE TABLE IF NOT EXISTS rewards (
    decision_id VARCHAR PRIMARY KEY,
    reward DOUBLE NOT NULL,
    n_reports INTEGER NOT NULL,
    n_sources INTEGER NOT NULL,
    weight DOUBLE NOT NULL,
    updated_at DOUBLE NOT NULL
)
"""
_META_DDL = "CREATE TABLE IF NOT EXISTS store_meta (key VARCHAR PRIMARY KEY, value VARCHAR)"


@dataclass(frozen=True, slots=True)
class LoggedRows:
    """An OPE batch plus the per-request sizes needed to price arms that were not taken."""

    batch: LoggedBatch
    input_tokens: IntArray
    max_output_tokens: IntArray
    decision_ids: tuple[str, ...]


class DecisionRecord(BaseModel):
    """One logged routing decision. Deliberately has no field that could carry a prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: Identifier
    logged_at: float
    tenant_id: Identifier
    policy_version: str = Field(min_length=1, max_length=128)
    policy_step: int = Field(ge=0)
    feature_version: str = Field(min_length=1, max_length=128)
    context_digest: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    arm: Identifier
    arm_index: int = Field(ge=0)
    propensity: float = Field(gt=0.0, le=1.0)
    probabilities: tuple[float, ...] = Field(min_length=2)
    features: tuple[float, ...] = Field(min_length=1)
    input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    estimated_cost_micros: int = Field(ge=0)
    ceiling_cost_micros: int = Field(ge=0)
    actual_cost_micros: int | None = Field(default=None, ge=0)
    workload_digest: str | None = None
    simulator_seed: int | None = None
    breaker_adjusted: bool = False
    served_by_fallback: bool = False

    def row(self) -> list[Any]:
        return [
            self.decision_id,
            self.logged_at,
            self.tenant_id,
            self.policy_version,
            self.policy_step,
            self.feature_version,
            self.context_digest,
            self.arm,
            self.arm_index,
            self.propensity,
            list(self.probabilities),
            list(self.features),
            self.input_tokens,
            self.max_output_tokens,
            self.estimated_cost_micros,
            self.ceiling_cost_micros,
            self.actual_cost_micros,
            self.workload_digest,
            self.simulator_seed,
            self.breaker_adjusted,
            self.served_by_fallback,
        ]


def _is_prompt_like(column: str) -> bool:
    """Whether a column name could plausibly hold request text.

    Whole tokens must match, except for ``prompt``, which never legitimately appears in this
    schema and is therefore also caught as a substring (``prompttext``, ``userprompt``).
    """
    lowered = column.lower()
    if "prompt" in lowered:
        return True
    tokens = {token for token in _COLUMN_TOKEN_SPLIT.split(lowered) if token}
    return bool(tokens & _FORBIDDEN_COLUMN_TOKENS)


class RouterLogStore:
    """DuckDB-backed decision, feedback and reward log. Safe for use from several threads."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        import duckdb  # noqa: PLC0415 - keeps CLI start-up light

        self._lock = threading.Lock()
        self._path = str(path)
        self._connection = duckdb.connect(self._path)
        with self._lock:
            for statement in (_DECISIONS_DDL, _FEEDBACK_DDL, _REWARDS_DDL, _META_DDL):
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT OR REPLACE INTO store_meta VALUES ('schema_version', ?)",
                [str(SCHEMA_VERSION)],
            )
            self._check_no_prompt_columns()

    # --- lifecycle ------------------------------------------------------------------------

    def _check_no_prompt_columns(self) -> None:
        rows = self._connection.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'main'"
        ).fetchall()
        offenders = [f"{table}.{column}" for table, column in rows if _is_prompt_like(str(column))]
        if offenders:
            raise RuntimeError(
                "the decision log must not contain prompt-like columns: " + ", ".join(offenders)
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> RouterLogStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def path(self) -> str:
        return self._path

    def columns(self, table: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    # --- writes ---------------------------------------------------------------------------

    def record_decision(self, record: DecisionRecord) -> None:
        self.record_decisions([record])

    def record_decisions(self, records: Sequence[DecisionRecord]) -> None:
        if not records:
            return
        placeholders = ", ".join(["?"] * len(records[0].row()))
        with self._lock:
            self._connection.executemany(
                # `placeholders` is "?" repeated len(row()) times; every value is bound as a
                # parameter below, so no caller text reaches the statement.
                f"INSERT INTO decisions VALUES ({placeholders})",  # noqa: S608  # nosec B608
                [record.row() for record in records],
            )

    def set_actual_cost(self, decision_id: str, actual_cost_micros: int) -> None:
        if actual_cost_micros < 0:
            raise ValueError("actual_cost_micros must be non-negative")
        with self._lock:
            self._connection.execute(
                "UPDATE decisions SET actual_cost_micros = ? WHERE decision_id = ?",
                [actual_cost_micros, decision_id],
            )

    # --- FeedbackStore protocol -----------------------------------------------------------

    def decision_timestamp(self, decision_id: str) -> float | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT logged_at FROM decisions WHERE decision_id = ?", [decision_id]
            ).fetchone()
        return float(row[0]) if row is not None else None

    def has_report(self, report_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM feedback WHERE report_id = ?", [report_id]
            ).fetchone()
        return row is not None

    def reports_for(self, decision_id: str) -> tuple[FeedbackRow, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT report_id, decision_id, source, raw_reward, reward, received_at "
                "FROM feedback WHERE decision_id = ? ORDER BY received_at, report_id",
                [decision_id],
            ).fetchall()
        return tuple(
            FeedbackRow(
                report_id=str(row[0]),
                decision_id=str(row[1]),
                source=str(row[2]),
                raw_reward=float(row[3]),
                reward=float(row[4]),
                received_at=float(row[5]),
            )
            for row in rows
        )

    def record_feedback(self, row: FeedbackRow) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                [
                    row.report_id,
                    row.decision_id,
                    row.source,
                    row.raw_reward,
                    row.reward,
                    row.received_at,
                ],
            )

    def set_reward(self, decision_id: str, aggregated: AggregatedReward) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO rewards VALUES (?, ?, ?, ?, ?, ?)",
                [
                    decision_id,
                    aggregated.reward,
                    aggregated.n_reports,
                    aggregated.n_sources,
                    aggregated.weight,
                    time.time(),
                ],
            )

    # --- reads ----------------------------------------------------------------------------

    def count_decisions(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT count(*) FROM decisions").fetchone()
        return int(row[0]) if row else 0

    def count_rewards(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT count(*) FROM rewards").fetchone()
        return int(row[0]) if row else 0

    def decision(self, decision_id: str) -> DecisionRecord | None:
        names = self.columns("decisions")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", [decision_id]
            ).fetchone()
        if row is None:
            return None
        payload = dict(zip(names, row, strict=True))
        payload["probabilities"] = tuple(float(value) for value in payload["probabilities"])
        payload["features"] = tuple(float(value) for value in payload["features"])
        return DecisionRecord.model_validate(payload)

    def arm_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT arm, count(*) FROM decisions GROUP BY arm ORDER BY arm"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def spend_by_tenant(self) -> dict[str, int]:
        """Committed spend per tenant in micro-USD (actual where known, else the estimate)."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT tenant_id, sum(coalesce(actual_cost_micros, estimated_cost_micros)) "
                "FROM decisions GROUP BY tenant_id ORDER BY tenant_id"
            ).fetchall()
        return {str(row[0]): int(row[1] or 0) for row in rows}

    def feature_matrix(self, *, limit: int | None = None, newest_first: bool = True) -> FloatArray:
        """Logged feature rows, for drift monitoring (PSI/KS against a reference window)."""
        # DuckDB cannot bind ORDER BY direction or LIMIT as parameters, so both are interpolated.
        # Neither can carry caller text: `order` is one of two literals chosen from a bool, and
        # `limit` is range-checked and passed through int() before interpolation.
        order = "DESC" if newest_first else "ASC"
        ordering = f"ORDER BY logged_at {order}, decision_id {order}"
        query = f"SELECT features FROM decisions {ordering}"  # noqa: S608  # nosec B608
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            query += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._connection.execute(query).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float64)
        return np.asarray([[float(value) for value in row[0]] for row in rows], dtype=np.float64)

    def logged_batch(
        self,
        *,
        n_arms: int,
        policy_versions: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> LoggedBatch:
        """Just the OPE batch; see :meth:`fetch_logged_rows` for the request sizes as well."""
        return self.fetch_logged_rows(
            n_arms=n_arms, policy_versions=policy_versions, limit=limit
        ).batch

    def fetch_logged_rows(
        self,
        *,
        n_arms: int,
        policy_versions: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> LoggedRows:
        """Join decisions with their aggregated rewards into an OPE batch plus request sizes.

        Rows without a reward are excluded: an importance-weighted estimator needs an outcome. The
        input and output token counts come back too, because screening a *different* policy needs
        the cost of arms that were not taken, and those are recomputed from the catalog rather than
        stored per arm.
        """
        clauses = ["r.reward IS NOT NULL"]
        params: list[Any] = []
        versions = list(policy_versions) if policy_versions is not None else None
        if versions:
            placeholders = ", ".join(["?"] * len(versions))
            clauses.append(f"d.policy_version IN ({placeholders})")
            params.extend(versions)
        where = " AND ".join(clauses)
        # `where` is built only from fixed literals plus `?` placeholders; every caller
        # value travels in `params`.
        select = (
            "SELECT d.features, d.arm_index, d.propensity, r.reward, d.input_tokens, "
            "d.max_output_tokens, d.decision_id "
            "FROM decisions d JOIN rewards r USING (decision_id)"
        )
        query = f"{select} WHERE {where} ORDER BY d.logged_at, d.decision_id"
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            query += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        if not rows:
            raise ValueError("no rewarded decisions are available for evaluation")
        features = np.asarray([[float(v) for v in row[0]] for row in rows], dtype=np.float64)
        actions = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
        propensities = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
        rewards = np.asarray([float(row[3]) for row in rows], dtype=np.float64)
        return LoggedRows(
            batch=LoggedBatch(
                features=features,
                actions=actions,
                propensities=propensities,
                rewards=rewards,
                n_arms=n_arms,
            ),
            input_tokens=np.asarray([int(row[4]) for row in rows], dtype=np.int64),
            max_output_tokens=np.asarray([int(row[5]) for row in rows], dtype=np.int64),
            decision_ids=tuple(str(row[6]) for row in rows),
        )
