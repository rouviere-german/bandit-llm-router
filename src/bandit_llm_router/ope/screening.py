"""Offline screening: what a candidate policy must prove before anyone may activate it.

Nothing about a candidate reaches traffic because it looked good in a notebook. A candidate is
evaluated on logged decisions and must clear three independent gates:

1. **It has to beat the incumbent counterfactually.** The comparison uses the *lower* end of the
   candidate's doubly robust bootstrap interval against the incumbent's DR point estimate, so a
   candidate whose apparent gain is inside its own noise is rejected. ``min_improvement`` can demand
   a margin on top.
2. **The log has to support the comparison.** Effective sample size must reach ``min_ess``. A
   candidate that concentrates on actions the log almost never took gets rejected for lack of
   evidence rather than promoted on a handful of highly weighted rows.
3. **Its worst case has to fit the budget.** Exploration is priced: for every logged context, the
   most expensive arm the candidate might pick is charged in full, extrapolated to
   ``horizon_requests``. If that bound exceeds ``max_exploration_usd`` the candidate is rejected
   however good its estimated value is.

A report is a record, not a recommendation: it carries the candidate's snapshot digest and the
content digest of the exact log it was computed on. :mod:`bandit_llm_router.activation` will only
activate a version whose approved report digest matches, so nobody can approve a report computed on
one log and activate against another.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from bandit_llm_router.core.canonical import canonical_json, sha256_hex
from bandit_llm_router.ope.estimators import (
    DEFAULT_BOOTSTRAP,
    CrossFitRidgeRewardModel,
    Estimate,
    LoggedBatch,
    OPEReport,
    evaluate_policy,
)

__all__ = [
    "CostBound",
    "EstimateRecord",
    "ScreeningConfig",
    "ScreeningReport",
    "TargetPolicy",
    "screen_candidate",
]

FloatArray = npt.NDArray[np.float64]
_MICROS_PER_USD = 1_000_000.0


class TargetPolicy(Protocol):
    """What screening needs from a policy: a version and batch action probabilities."""

    name: str

    @property
    def version(self) -> str: ...

    def probabilities_matrix(
        self, features: FloatArray, rng: np.random.Generator
    ) -> FloatArray: ...


class ScreeningConfig(BaseModel):
    """Thresholds a candidate must clear. These are governance settings, so they are versioned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_ess: float = Field(default=500.0, gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    bootstrap: int = Field(default=DEFAULT_BOOTSTRAP, ge=0, le=10_000)
    folds: int = Field(default=4, ge=2, le=20)
    ridge: float = Field(default=1.0, gt=0.0)
    weight_clip: float | None = Field(default=None, gt=0.0)
    min_improvement: float = Field(default=0.0, ge=0.0, le=1.0)
    horizon_requests: int = Field(default=100_000, ge=1)
    max_exploration_usd: float = Field(default=250.0, gt=0.0)

    def digest_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class EstimateRecord(BaseModel):
    """JSON-safe copy of an :class:`~bandit_llm_router.ope.estimators.Estimate`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    ess: float
    clipped_fraction: float

    @classmethod
    def of(cls, estimate: Estimate) -> EstimateRecord:
        return cls(
            method=estimate.method,
            value=estimate.value,
            std_error=estimate.std_error,
            ci_low=estimate.ci_low,
            ci_high=estimate.ci_high,
            ess=estimate.ess,
            clipped_fraction=estimate.clipped_fraction,
        )


class CostBound(BaseModel):
    """What the candidate would cost per request, expected and worst case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_usd_per_request: float = Field(ge=0.0)
    worst_case_usd_per_request: float = Field(ge=0.0)
    worst_case_horizon_usd: float = Field(ge=0.0)


class ScreeningReport(BaseModel):
    """The full record of one screening decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: str
    candidate_version: str
    candidate_name: str
    incumbent_version: str
    logs_digest: str
    config: ScreeningConfig
    n: int = Field(ge=1)
    ess: float = Field(ge=0.0)
    min_propensity: float = Field(gt=0.0)
    mean_logged_reward: float
    candidate_ips: EstimateRecord
    candidate_snips: EstimateRecord
    candidate_dr: EstimateRecord
    incumbent_dr: EstimateRecord
    dr_lower_bound: float
    improvement: float
    cost: CostBound
    eligible: bool
    reasons: tuple[str, ...] = ()

    def digest(self) -> str:
        """Content hash of the whole report; activation binds to this value."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def summary_row(self) -> dict[str, object]:
        return {
            "candidate": self.candidate_version,
            "dr": round(self.candidate_dr.value, 4),
            "dr_lower": round(self.dr_lower_bound, 4),
            "incumbent_dr": round(self.incumbent_dr.value, 4),
            "ess": round(self.ess, 1),
            "worst_case_usd": round(self.cost.worst_case_horizon_usd, 2),
            "eligible": self.eligible,
            "reasons": ",".join(self.reasons) or "-",
        }


def _cost_bound(
    probabilities: FloatArray,
    expected_cost_micros: FloatArray,
    ceiling_cost_micros: FloatArray,
    *,
    horizon: int,
) -> CostBound:
    expected = float(np.einsum("ij,ij->i", probabilities, expected_cost_micros).mean())
    reachable = probabilities > 0.0
    masked = np.where(reachable, ceiling_cost_micros, -np.inf)
    worst = float(np.max(masked, axis=1).mean())
    return CostBound(
        expected_usd_per_request=expected / _MICROS_PER_USD,
        worst_case_usd_per_request=worst / _MICROS_PER_USD,
        worst_case_horizon_usd=worst * horizon / _MICROS_PER_USD,
    )


def screen_candidate(
    batch: LoggedBatch,
    *,
    candidate: TargetPolicy,
    incumbent: TargetPolicy,
    expected_cost_micros: FloatArray,
    ceiling_cost_micros: FloatArray,
    config: ScreeningConfig | None = None,
    rng: np.random.Generator | None = None,
    clock: datetime | None = None,
) -> ScreeningReport:
    """Evaluate ``candidate`` against ``incumbent`` on ``batch`` and decide eligibility."""
    settings = config if config is not None else ScreeningConfig()
    generator = rng if rng is not None else np.random.default_rng(0)
    if expected_cost_micros.shape != (batch.n, batch.n_arms):
        raise ValueError("expected_cost_micros must have shape (n, n_arms)")
    if ceiling_cost_micros.shape != (batch.n, batch.n_arms):
        raise ValueError("ceiling_cost_micros must have shape (n, n_arms)")
    if np.any(expected_cost_micros < 0.0) or np.any(ceiling_cost_micros < 0.0):
        raise ValueError("costs must be non-negative")

    candidate_probabilities = candidate.probabilities_matrix(
        batch.features, np.random.default_rng(generator.integers(0, 2**62))
    )
    incumbent_probabilities = incumbent.probabilities_matrix(
        batch.features, np.random.default_rng(generator.integers(0, 2**62))
    )
    # One cross-fitted reward model for both sides: the comparison is then between policies, not
    # between two differently fitted nuisance models.
    q_hat = CrossFitRidgeRewardModel(ridge=settings.ridge, folds=settings.folds).fit_predict(
        batch, generator
    )
    candidate_report: OPEReport = evaluate_policy(
        batch,
        candidate_probabilities,
        clip=settings.weight_clip,
        bootstrap=settings.bootstrap,
        alpha=settings.alpha,
        rng=generator,
        q_hat=q_hat,
    )
    incumbent_report: OPEReport = evaluate_policy(
        batch,
        incumbent_probabilities,
        clip=settings.weight_clip,
        bootstrap=settings.bootstrap,
        alpha=settings.alpha,
        rng=generator,
        q_hat=q_hat,
    )
    cost = _cost_bound(
        candidate_probabilities,
        expected_cost_micros,
        ceiling_cost_micros,
        horizon=settings.horizon_requests,
    )
    dr_lower = candidate_report.dr.ci_low
    improvement = dr_lower - incumbent_report.dr.value
    reasons: list[str] = []
    if improvement < settings.min_improvement:
        reasons.append("dr_lower_bound_does_not_beat_incumbent")
    if candidate_report.dr.ess < settings.min_ess:
        reasons.append("effective_sample_size_below_minimum")
    if cost.worst_case_horizon_usd > settings.max_exploration_usd:
        reasons.append("worst_case_cost_exceeds_budget")
    return ScreeningReport(
        created_at=(clock or datetime.now(UTC)).isoformat(),
        candidate_version=candidate.version,
        candidate_name=candidate.name,
        incumbent_version=incumbent.version,
        logs_digest=batch.digest,
        config=settings,
        n=batch.n,
        ess=candidate_report.dr.ess,
        min_propensity=candidate_report.min_propensity,
        mean_logged_reward=candidate_report.mean_logged_reward,
        candidate_ips=EstimateRecord.of(candidate_report.ips),
        candidate_snips=EstimateRecord.of(candidate_report.snips),
        candidate_dr=EstimateRecord.of(candidate_report.dr),
        incumbent_dr=EstimateRecord.of(incumbent_report.dr),
        dr_lower_bound=dr_lower,
        improvement=improvement,
        cost=cost,
        eligible=not reasons,
        reasons=tuple(reasons),
    )
