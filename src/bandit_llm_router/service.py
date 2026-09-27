"""The routing service: decide, spend, serve, log, learn, and gate policy activation.

One request flows through this file end to end:

1. **Validate** into a :class:`~bandit_llm_router.features.RoutingContext` (bounded prompt, bounded
   output request, known tenant shape) and featurize it. The prompt goes no further than the
   featurizer.
2. **Decide** with the live policy, which returns a distribution over arms, never a single arm.
3. **Restrict** that distribution: arms whose breaker is open, and arms whose worst-case cost does
   not fit the tenant's remaining budget, are removed and their probability mass moves to the first
   affordable, healthy arm in the catalog's fallback order. What gets logged is the propensity of
   the distribution that actually ran.
4. **Reserve** the chosen arm's cost ceiling against the tenant's cap *before* calling the provider,
   then call it. A provider failure releases the reservation, trips the breaker and redirects once
   to the fallback arm, and the redirected propensity is what is logged.
5. **Commit** the real cost from the response's usage, log the decision (features and digest, no
   prompt) and update the metrics.

Rewards arrive later on ``/v1/feedback`` (a different key scope), go through the poisoning-resistant
ingestor, and update the live posterior once per decision -- on the first accepted report. Later
reports refine the stored aggregate, which is what off-policy evaluation and offline retraining use;
they are deliberately not re-applied to the online posterior, because an online ridge update cannot
be withdrawn and applying several would double-count one outcome.

Policy promotion lives here too, but the service only *runs* the gate:
``/v1/policies/{digest}/screen`` computes a screening report from the service's own logs (so the
evidence cannot be supplied by the caller), ``propose`` refuses anything the report did not clear,
``approve`` must come from a different principal, and ``activate`` re-checks the evidence before
moving the ``active`` alias. Every step is appended to a hash-chained audit log.

The gate's statistical caveat, stated where it happens: the log the service screens on was written
by the live policy *while that policy was learning*, so its rows are serially dependent. Two things
follow. The bootstrap behind ``dr_lower_bound`` resamples contiguous blocks of rows rather than
single rows (``screening_bootstrap_block_size``), and :meth:`RouterService.screen` accepts a
``policy_step`` window so an operator can screen on a stretch of the log over which the posterior
barely moved. Neither makes an adaptively collected log i.i.d.; the interval coverage measured in
``bench/bench_ope.py`` on a *frozen* logging policy does not transfer unchanged to this path, and
``bench/bench_ope.py`` measures the gap rather than assuming it away (ADR 0008).

One honest caveat about the fallback propensity: redirecting after a failure assumes a provider
failure is a property of the arm at that moment, independent of the request. If an arm failed
preferentially on particular contexts, the adjusted propensity would be only approximately right.
ADR 0005 records the alternatives.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import Depends, FastAPI, Query
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bandit_llm_router.activation import (
    POLICY_MODEL_NAME,
    ActivationError,
    ActivationRecord,
    NotEligibleError,
    PolicyActivationController,
    SeparationOfDutiesError,
    UnknownActivationError,
)
from bandit_llm_router.agentic.audit import AuditLog
from bandit_llm_router.agentic.errors import ProviderError
from bandit_llm_router.agentic.providers.base import ChatMessage, LLMProvider, LLMRequest
from bandit_llm_router.arms import ArmCatalog, load_catalog
from bandit_llm_router.auth import Scope, ScopedApiKeyVerifier, ScopedPrincipal, require_scope
from bandit_llm_router.breaker import BreakerConfig, CircuitBreaker
from bandit_llm_router.budgets import (
    BudgetExhaustedError,
    TenantBudgetGuard,
    TenantBudgets,
)
from bandit_llm_router.core.app import ReadinessCheck, create_app
from bandit_llm_router.core.config import Settings, get_settings
from bandit_llm_router.core.errors import AppError
from bandit_llm_router.core.logging import get_logger
from bandit_llm_router.core.ratelimit import RateLimiter
from bandit_llm_router.features import MAX_PROMPT_CHARS, Featurizer, RoutingContext
from bandit_llm_router.feedback import (
    DuplicateReportError,
    ExpiredDecisionError,
    FeedbackError,
    FeedbackIngestor,
    FeedbackPolicy,
    FeedbackRateLimitedError,
    FeedbackSubmission,
    TooManyReportsError,
    UnknownDecisionError,
)
from bandit_llm_router.logstore import DecisionRecord, RouterLogStore
from bandit_llm_router.metrics import RouterMetrics, create_router_metrics
from bandit_llm_router.ml.registry import ModelRegistry
from bandit_llm_router.ope.screening import ScreeningConfig, ScreeningReport, screen_candidate
from bandit_llm_router.policies.base import (
    NoArmAvailableError,
    restrict_distribution,
    sample_action,
)
from bandit_llm_router.policies.factory import (
    PolicyHyperparameters,
    build_policy,
    policy_from_registry_files,
)
from bandit_llm_router.policies.linear_bandit import LinearBanditPolicy
from bandit_llm_router.provider import build_provider

__all__ = [
    "FeedbackResponse",
    "PolicyStatus",
    "RouteRequest",
    "RouteResponse",
    "RouterService",
    "build_router_service",
    "create_router_app",
]

FloatArray = npt.NDArray[np.float64]
_log = get_logger("bandit_llm_router.service")
RoutePrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.ROUTE))]
FeedbackPrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.FEEDBACK))]
AdminPrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.ADMIN))]
ApprovePrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.APPROVE))]
_MICROS_PER_USD = 1_000_000.0
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


# --- API models ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """A routing request. The prompt is featurized and dropped; it is never stored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")]
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_output_tokens: int = Field(default=512, ge=1, le=8_192)
    tier: Literal["standard", "priority"] = "standard"

    def context(self) -> RoutingContext:
        return RoutingContext(
            tenant_id=self.tenant_id,
            prompt=self.prompt,
            max_output_tokens=self.max_output_tokens,
            tier=self.tier,
        )


class RouteResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    arm: str
    model: str
    policy_version: str
    propensity: float
    estimated_cost_usd: float
    committed_cost_usd: float
    served_by_fallback: bool
    text: str


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    reward: float
    n_reports: int
    n_sources: int
    weight: float
    clipped: bool
    applied_to_policy: bool


class PolicyStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    live_policy_version: str
    live_policy_updates: int
    active_snapshot_digest: str | None
    registered_versions: tuple[str, ...]
    breaker: Mapping[str, str]
    decisions_logged: int
    rewards_logged: int
    arm_counts: Mapping[str, int]
    audit_head: str


# --- service ------------------------------------------------------------------------------


@dataclass(slots=True)
class RouterService:
    """Everything a routing decision needs, wired together and free of framework details."""

    settings: Settings
    catalog: ArmCatalog
    featurizer: Featurizer
    policy: LinearBanditPolicy
    store: RouterLogStore
    guard: TenantBudgetGuard
    breaker: CircuitBreaker
    ingestor: FeedbackIngestor
    metrics: RouterMetrics
    provider: LLMProvider
    audit: AuditLog
    controller: PolicyActivationController
    registry: ModelRegistry
    screening: ScreeningConfig
    rng: np.random.Generator
    min_propensity_seen: float = 1.0
    # Where this instance's traffic comes from, written to every logged row. Both are None in a
    # deployment fed by real callers and set when a simulator or bench drives the service, which is
    # the only case where "which workload produced this row" has an answer the service can know.
    # The decision log is then self-describing: policy version, policy step, workload digest, seed.
    workload_digest: str | None = None
    simulator_seed: int | None = None

    # --- routing --------------------------------------------------------------------------

    def route(self, request: RouteRequest, *, principal: ScopedPrincipal) -> RouteResponse:
        """Pick an arm, serve the request and log the decision."""
        context = request.context()
        features = self.featurizer.transform(context)
        ceilings = self.catalog.cost_ceiling_vector(
            context.input_token_ceiling, context.max_output_tokens
        )
        expected = self.catalog.expected_cost_vector(features.estimated_input_tokens)
        breaker_blocked = self.breaker.blocked_mask()
        unaffordable = ~self.guard.affordable_mask(context.tenant_id, ceilings)
        distribution = self.policy.probabilities(features.vector, self.rng)
        blocked = breaker_blocked | unaffordable
        try:
            realized = restrict_distribution(distribution, blocked, self.catalog.fallback_order)
        except NoArmAvailableError as exc:
            self.metrics.budget_denied_total.labels(context.tenant_id).inc()
            _log.warning(
                "no_arm_available",
                tenant=context.tenant_id,
                breaker=self.breaker.snapshot(),
                key_id=principal.key_id,
            )
            raise AppError(
                402,
                "no_arm_available",
                "No routing arm is available within this tenant's remaining budget.",
            ) from exc
        arm_index = sample_action(realized, self.rng)
        adjusted = bool(blocked.any())
        served = self._serve(
            context=context,
            arm_index=arm_index,
            distribution=distribution,
            blocked=blocked,
            ceilings=ceilings,
        )
        propensity = float(served.realized[served.arm_index])
        arm = self.catalog.names[served.arm_index]
        spec = self.catalog.spec(served.arm_index)
        decision_id = uuid.uuid4().hex
        record = DecisionRecord(
            decision_id=decision_id,
            logged_at=time.time(),
            tenant_id=context.tenant_id,
            policy_version=self.policy.version,
            policy_step=self.policy.updates,
            feature_version=self.featurizer.version,
            context_digest=features.context_digest,
            arm=arm,
            arm_index=served.arm_index,
            propensity=propensity,
            probabilities=tuple(float(value) for value in served.realized),
            features=tuple(float(value) for value in features.vector),
            input_tokens=features.estimated_input_tokens,
            max_output_tokens=context.max_output_tokens,
            estimated_cost_micros=int(expected[served.arm_index]),
            ceiling_cost_micros=int(ceilings[served.arm_index]),
            actual_cost_micros=served.committed_micros,
            workload_digest=self.workload_digest,
            simulator_seed=self.simulator_seed,
            breaker_adjusted=adjusted or served.fallback_used,
            served_by_fallback=served.fallback_used,
        )
        self.store.record_decision(record)
        self._update_route_metrics(record, served)
        _log.info(
            "route_decision",
            key_id=principal.key_id,
            decision_id=decision_id,
            tenant=context.tenant_id,
            arm=arm,
            propensity=round(propensity, 6),
            committed_usd=round(served.committed_micros / _MICROS_PER_USD, 6),
            fallback=served.fallback_used,
        )
        return RouteResponse(
            decision_id=decision_id,
            arm=arm,
            model=spec.model,
            policy_version=self.policy.version,
            propensity=propensity,
            estimated_cost_usd=float(expected[served.arm_index]) / _MICROS_PER_USD,
            committed_cost_usd=served.committed_micros / _MICROS_PER_USD,
            served_by_fallback=served.fallback_used,
            text=served.text,
        )

    @dataclass(frozen=True, slots=True)
    class _Served:
        arm_index: int
        realized: FloatArray
        committed_micros: int
        text: str
        fallback_used: bool

    def _serve(
        self,
        *,
        context: RoutingContext,
        arm_index: int,
        distribution: FloatArray,
        blocked: npt.NDArray[np.bool_],
        ceilings: FloatArray,
    ) -> RouterService._Served:
        """Call the provider, redirecting once past an arm that fails."""
        failed = np.zeros(len(self.catalog), dtype=np.bool_)
        realized = restrict_distribution(distribution, blocked, self.catalog.fallback_order)
        fallback_used = False
        for _ in range(2):
            arm = self.catalog.names[arm_index]
            spec = self.catalog.spec(arm_index)
            try:
                reservation = self.guard.reserve(context.tenant_id, arm, int(ceilings[arm_index]))
            except BudgetExhaustedError as exc:
                self.metrics.budget_denied_total.labels(context.tenant_id).inc()
                raise AppError(
                    402, "budget_exhausted", "This tenant's budget cannot cover the request."
                ) from exc
            llm_request = LLMRequest(
                messages=(ChatMessage.user(context.prompt),),
                role=arm,
                max_tokens=min(context.max_output_tokens, spec.max_output_tokens),
            )
            try:
                response = self.provider.complete(llm_request)
            except ProviderError as exc:
                self.guard.release(reservation)
                self.breaker.record_failure(arm)
                self.metrics.provider_failures_total.labels(arm, type(exc).__name__).inc()
                failed[arm_index] = True
                unavailable = blocked | failed
                if bool(unavailable.all()):
                    raise AppError(
                        503, "provider_unavailable", "No routing arm could serve the request."
                    ) from exc
                realized = restrict_distribution(
                    distribution, unavailable, self.catalog.fallback_order
                )
                substitute = next(
                    index for index in self.catalog.fallback_order if not unavailable[index]
                )
                self.metrics.provider_fallbacks_total.labels(
                    arm, self.catalog.names[substitute]
                ).inc()
                _log.warning(
                    "provider_failed",
                    arm=arm,
                    error_type=type(exc).__name__,
                    fallback_arm=self.catalog.names[substitute],
                )
                arm_index = substitute
                fallback_used = True
                continue
            actual = spec.cost_micros(response.usage.input_tokens, response.usage.output_tokens)
            committed = self.guard.commit(reservation, actual)
            if actual > reservation.micros:
                _log.warning(
                    "budget_overshoot",
                    arm=arm,
                    reserved_micros=reservation.micros,
                    actual_micros=actual,
                )
            self.breaker.record_success(arm)
            return RouterService._Served(
                arm_index=arm_index,
                realized=realized,
                committed_micros=committed,
                text=response.text,
                fallback_used=fallback_used,
            )
        raise AppError(503, "provider_unavailable", "No routing arm could serve the request.")

    def _update_route_metrics(self, record: DecisionRecord, served: RouterService._Served) -> None:
        self.metrics.route_decisions_total.labels(record.arm, record.policy_version).inc()
        self.metrics.spend_usd_total.labels(record.tenant_id, record.arm).inc(
            served.committed_micros / _MICROS_PER_USD
        )
        self.min_propensity_seen = min(self.min_propensity_seen, record.propensity)
        self.metrics.propensity_min.set(self.min_propensity_seen)
        for arm, state in self.breaker.snapshot().items():
            self.metrics.breaker_state.labels(arm).set(
                {"closed": 0.0, "half_open": 1.0, "open": 2.0}[state]
            )

    # --- feedback -------------------------------------------------------------------------

    def submit_feedback(
        self, submission: FeedbackSubmission, *, principal: ScopedPrincipal
    ) -> FeedbackResponse:
        """Ingest one reward report and, on the first report for a decision, teach the policy."""
        try:
            result = self.ingestor.submit(submission, source=principal.key_id)
        except FeedbackError as exc:
            reason = type(exc).__name__
            self.metrics.feedback_rejected_total.labels(reason).inc()
            status, code = _FEEDBACK_ERRORS.get(type(exc), (400, "feedback_rejected"))
            raise AppError(status, code, "The feedback report was refused.") from exc
        if result.clipped:
            self.metrics.feedback_clipped_total.inc()
        applied = False
        decision = self.store.decision(submission.decision_id)
        if (
            decision is not None
            and decision.policy_version == self.policy.version
            and result.aggregated.n_reports == 1
        ):
            self.policy.update(
                np.asarray(decision.features, dtype=np.float64),
                decision.arm_index,
                result.aggregated.reward,
                result.aggregated.weight,
            )
            applied = True
        if decision is not None:
            self.metrics.estimated_quality.labels(decision.arm).observe(result.aggregated.reward)
        _log.info(
            "feedback_ingested",
            key_id=principal.key_id,
            decision_id=submission.decision_id,
            reward=round(result.aggregated.reward, 4),
            sources=result.aggregated.n_sources,
            applied=applied,
        )
        return FeedbackResponse(
            decision_id=submission.decision_id,
            reward=result.aggregated.reward,
            n_reports=result.aggregated.n_reports,
            n_sources=result.aggregated.n_sources,
            weight=result.aggregated.weight,
            clipped=result.clipped,
            applied_to_policy=applied,
        )

    # --- policy lifecycle -----------------------------------------------------------------

    def status(self) -> PolicyStatus:
        active = self.controller.active_version()
        return PolicyStatus(
            live_policy_version=self.policy.version,
            live_policy_updates=self.policy.updates,
            active_snapshot_digest=active.digest if active is not None else None,
            registered_versions=tuple(self.registry.versions(POLICY_MODEL_NAME)),
            breaker=self.breaker.snapshot(),
            decisions_logged=self.store.count_decisions(),
            rewards_logged=self.store.count_rewards(),
            arm_counts=self.store.arm_counts(),
            audit_head=self.audit.head,
        )

    def screen(
        self,
        snapshot_digest: str,
        *,
        limit: int | None = None,
        min_policy_step: int | None = None,
        max_policy_step: int | None = None,
    ) -> ScreeningReport:
        """Screen a registered snapshot against the live policy on the service's own logs.

        ``min_policy_step`` / ``max_policy_step`` narrow the evidence to a window of the live
        policy's update counter, which is how an operator asks for a stretch of the log over which
        the logging policy was nearly frozen. See the module docstring for why that matters.
        """
        candidate = self._candidate_from_registry(snapshot_digest)
        try:
            rows = self.store.fetch_logged_rows(
                n_arms=len(self.catalog),
                min_policy_step=min_policy_step,
                max_policy_step=max_policy_step,
                limit=limit,
            )
        except ValueError as exc:
            raise AppError(
                409, "insufficient_evidence", "No rewarded decisions match that log window."
            ) from exc
        expected, ceiling = self.catalog.cost_matrices(rows.input_tokens, rows.max_output_tokens)
        report = screen_candidate(
            rows.batch,
            candidate=candidate,
            incumbent=self.policy,
            expected_cost_micros=expected,
            ceiling_cost_micros=ceiling,
            config=self.screening,
            rng=self.rng,
        )
        self.metrics.ope_dr_lower_bound.labels(report.candidate_version).set(report.dr_lower_bound)
        self.metrics.ope_ess.labels(report.candidate_version).set(report.ess)
        return report

    def _candidate_from_registry(self, snapshot_digest: str) -> LinearBanditPolicy:
        try:
            version = self.registry.get(POLICY_MODEL_NAME, snapshot_digest)
            files = self.registry.load_files(version)
        except ValueError as exc:
            raise AppError(404, "not_found", "No such policy snapshot.") from exc
        candidate = policy_from_registry_files(files)
        if candidate.dim != self.policy.dim or candidate.arms != self.policy.arms:
            raise AppError(
                409, "incompatible_policy", "The snapshot does not match the live arm set."
            )
        return candidate

    def propose(
        self, snapshot_digest: str, *, principal: ScopedPrincipal, limit: int | None = None
    ) -> ActivationRecord:
        report = self.screen(snapshot_digest, limit=limit)
        try:
            record = self.controller.propose(
                snapshot_digest=snapshot_digest, report=report, proposed_by=principal.key_id
            )
        except NotEligibleError as exc:
            self.metrics.policy_activations_total.labels("screening_rejected").inc()
            raise AppError(
                409, "screening_rejected", "The candidate did not clear OPE screening."
            ) from exc
        except ActivationError as exc:
            raise AppError(
                409, "activation_conflict", "The activation could not be proposed."
            ) from exc
        self.metrics.policy_activations_total.labels("proposed").inc()
        return record

    def approve(self, activation_id: str, *, principal: ScopedPrincipal) -> ActivationRecord:
        try:
            record = self.controller.approve(activation_id, approver=principal.key_id)
        except SeparationOfDutiesError as exc:
            self.metrics.policy_activations_total.labels("self_approval_refused").inc()
            raise AppError(
                403, "separation_of_duties", "The proposer cannot approve their own activation."
            ) from exc
        except UnknownActivationError as exc:
            raise AppError(404, "not_found", "No such activation.") from exc
        except ActivationError as exc:
            raise AppError(
                409, "activation_conflict", "The activation cannot be approved."
            ) from exc
        self.metrics.policy_activations_total.labels("approved").inc()
        return record

    def activate(self, activation_id: str, *, principal: ScopedPrincipal) -> ActivationRecord:
        try:
            record = self.controller.activate(activation_id, actor=principal.key_id)
        except UnknownActivationError as exc:
            raise AppError(404, "not_found", "No such activation.") from exc
        except ActivationError as exc:
            self.metrics.policy_activations_total.labels("refused").inc()
            raise AppError(409, "activation_conflict", "The activation cannot proceed.") from exc
        candidate = self._candidate_from_registry(record.snapshot_digest)
        self.policy = candidate
        self.metrics.policy_activations_total.labels("activated").inc()
        _log.info(
            "policy_activated",
            key_id=principal.key_id,
            activation_id=record.id,
            policy_version=record.candidate_version,
            audit_head=self.audit.head,
        )
        return record

    def register_live_policy(self, *, metrics: Mapping[str, float] | None = None) -> str:
        """Snapshot the live policy into the registry and return its content digest."""
        version = self.controller.register_snapshot(self.policy.snapshot(), metrics=metrics)
        return version.digest

    # --- readiness ------------------------------------------------------------------------

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        """Readiness probes: a policy is loaded, an arm is usable, and the log store answers."""
        return {
            "policy_loaded": lambda: bool(self.policy.version),
            "arm_available": self.breaker.any_available,
            "log_store": lambda: self.store.count_decisions() >= 0,
        }


_FEEDBACK_ERRORS: Mapping[type[FeedbackError], tuple[int, str]] = {
    UnknownDecisionError: (404, "unknown_decision"),
    ExpiredDecisionError: (409, "feedback_window_closed"),
    DuplicateReportError: (409, "duplicate_report"),
    TooManyReportsError: (429, "too_many_reports"),
    FeedbackRateLimitedError: (429, "rate_limited"),
}


# --- wiring -------------------------------------------------------------------------------


def build_router_service(
    settings: Settings | None = None,
    *,
    registry: CollectorRegistry | None = None,
    provider: LLMProvider | None = None,
    catalog: ArmCatalog | None = None,
    seed: int = 0,
    workload_digest: str | None = None,
    simulator_seed: int | None = None,
) -> RouterService:
    """Wire a service from settings, creating the data directories it needs.

    ``workload_digest`` and ``simulator_seed`` identify the traffic source and are written to every
    logged decision. They are meaningful only when a simulator drives the service (the demo and
    ``bench/bench_router.py`` pass their own); under real callers they stay ``None``, because the
    service cannot invent an answer.
    """
    cfg = settings if settings is not None else get_settings()
    arms = catalog if catalog is not None else load_catalog(cfg.pricing_path)
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    digest_key = (
        cfg.feature_digest_key.get_secret_value().encode("utf-8")
        if cfg.feature_digest_key is not None
        else b""
    )
    featurizer = Featurizer(digest_key=digest_key)
    policy = build_policy(
        PolicyHyperparameters(
            kind=cfg.policy_kind,
            alpha=cfg.policy_alpha,
            epsilon=cfg.policy_epsilon,
            posterior_sd=cfg.policy_posterior_sd,
            mc_samples=cfg.policy_mc_samples,
            ridge=cfg.policy_ridge,
            discount=cfg.policy_discount,
            propensity_floor=cfg.propensity_floor,
        ),
        arms=arms.names,
        dim=featurizer.dim,
        feature_version=featurizer.version,
    )
    store = RouterLogStore(data_dir / "decisions.duckdb")
    guard = TenantBudgetGuard(
        TenantBudgets(
            caps_usd=dict(cfg.tenant_budget_caps),
            default_usd=cfg.tenant_budget_usd,
            window_seconds=cfg.tenant_budget_window_seconds,
        ),
        path=data_dir / "budgets.json",
    )
    breaker = CircuitBreaker(
        arms.names,
        config=BreakerConfig(
            failure_threshold=cfg.breaker_failure_threshold,
            cooldown_seconds=cfg.breaker_cooldown_seconds,
        ),
    )
    ingestor = FeedbackIngestor(
        store,
        policy=FeedbackPolicy(
            ttl_seconds=cfg.feedback_ttl_seconds,
            max_reports_per_decision=cfg.feedback_max_reports_per_decision,
        ),
        limiter=RateLimiter(cfg.feedback_rate_limit_per_second, cfg.feedback_rate_limit_burst),
    )
    audit = AuditLog(data_dir / "activations.audit.jsonl")
    model_registry = ModelRegistry(data_dir / "registry")
    controller = PolicyActivationController(
        registry=model_registry, audit=audit, path=data_dir / "activations.json"
    )
    return RouterService(
        settings=cfg,
        catalog=arms,
        featurizer=featurizer,
        policy=policy,
        store=store,
        guard=guard,
        breaker=breaker,
        ingestor=ingestor,
        metrics=create_router_metrics(registry if registry is not None else CollectorRegistry()),
        provider=provider if provider is not None else build_provider(cfg, arms, seed=seed),
        audit=audit,
        controller=controller,
        registry=model_registry,
        screening=ScreeningConfig(
            min_ess=cfg.screening_min_ess,
            horizon_requests=cfg.screening_horizon_requests,
            max_exploration_usd=cfg.screening_max_exploration_usd,
            bootstrap_block_size=cfg.screening_bootstrap_block_size,
        ),
        rng=np.random.default_rng(seed),
        workload_digest=workload_digest,
        simulator_seed=simulator_seed,
    )


def create_router_app(
    settings: Settings | None = None,
    *,
    service: RouterService | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the hardened base app and mount the router's endpoints on it."""
    cfg = settings if settings is not None else get_settings()
    metrics_registry = registry if registry is not None else CollectorRegistry()
    router_service = (
        service if service is not None else build_router_service(cfg, registry=metrics_registry)
    )
    scoped = ScopedApiKeyVerifier.from_settings(cfg)
    # The baseline app uses api_key_sha256 for /metrics auth and per-key rate limiting; every scoped
    # key must be known there too, so the union is passed down.
    base_settings = cfg.model_copy(
        update={"api_key_sha256": sorted({*cfg.api_key_sha256, *scoped.digests})}
    )
    app = create_app(
        base_settings,
        readiness_checks=router_service.readiness_checks(),
        registry=metrics_registry,
    )
    app.state.scoped_verifier = scoped
    app.state.router = router_service
    app.state.router_metrics = router_service.metrics

    @app.post("/v1/route", response_model=RouteResponse)
    async def route(body: RouteRequest, principal: RoutePrincipal) -> RouteResponse:
        """Route one request to an arm, serve it and log the decision with its propensity."""
        return router_service.route(body, principal=principal)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    async def feedback(body: FeedbackSubmission, principal: FeedbackPrincipal) -> FeedbackResponse:
        """Report a reward for a logged decision."""
        return router_service.submit_feedback(body, principal=principal)

    @app.get("/v1/policies", response_model=PolicyStatus)
    async def policies(principal: AdminPrincipal) -> PolicyStatus:
        """Live policy, registered snapshots, breaker state and log counts."""
        del principal
        return router_service.status()

    @app.post("/v1/policies/{snapshot_digest}/screen", response_model=ScreeningReport)
    async def screen(
        snapshot_digest: Digest,
        principal: AdminPrincipal,
        min_policy_step: Annotated[int | None, Query(ge=0)] = None,
        max_policy_step: Annotated[int | None, Query(ge=0)] = None,
    ) -> ScreeningReport:
        """Run OPE screening for a registered snapshot against the live policy."""
        del principal
        return router_service.screen(
            snapshot_digest, min_policy_step=min_policy_step, max_policy_step=max_policy_step
        )

    @app.post("/v1/policies/{snapshot_digest}/propose", response_model=ActivationRecord)
    async def propose(snapshot_digest: Digest, principal: AdminPrincipal) -> ActivationRecord:
        """Propose activating a snapshot; refused unless screening passed."""
        return router_service.propose(snapshot_digest, principal=principal)

    @app.get("/v1/activations", response_model=list[ActivationRecord])
    async def activations(principal: AdminPrincipal) -> list[ActivationRecord]:
        """Every activation request, oldest first."""
        del principal
        return list(router_service.controller.records())

    @app.post("/v1/activations/{activation_id}/approve", response_model=ActivationRecord)
    async def approve(activation_id: Identifier, principal: ApprovePrincipal) -> ActivationRecord:
        """Approve an activation (must not be the proposer)."""
        return router_service.approve(activation_id, principal=principal)

    @app.post("/v1/activations/{activation_id}/activate", response_model=ActivationRecord)
    async def activate(activation_id: Identifier, principal: AdminPrincipal) -> ActivationRecord:
        """Move the ``active`` alias to an approved snapshot and swap the live policy."""
        return router_service.activate(activation_id, principal=principal)

    return app
