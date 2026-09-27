"""Router metrics: what an on-call engineer needs to see that routing is behaving.

Beyond the baseline HTTP metrics, the router exports the numbers that make a *learned* decision
policy operable:

* ``route_decisions_total{arm}`` and ``provider_fallbacks_total{from_arm,to_arm}`` -- where traffic
  went, and where the breaker moved it.
* ``spend_usd_total{tenant}`` and ``budget_denied_total{tenant}`` -- money, and refusals.
* ``estimated_quality`` -- histogram of the quality signal behind the rewards.
* ``propensity_min`` -- the smallest propensity logged recently. If this drifts toward zero the log
  is quietly becoming unusable for off-policy evaluation, long before any estimate looks wrong.
* ``ope_dr_lower_bound``, ``ope_ess`` -- the last screening run's doubly robust lower bound and
  effective sample size, so a rejected candidate is visible without reading logs.
* ``context_feature_psi{feature}`` -- population stability of the logged context features against a
  reference window: the input-drift alarm for a policy that was trained on yesterday's traffic.
* ``feedback_rejected_total{reason}`` and ``feedback_clipped_total`` -- the shape of hostile or
  broken feedback.
* ``policy_activations_total{result}`` -- how often activation was attempted and what happened.

Metrics live on a per-app registry, never the global one, so tests stay isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["RouterMetrics", "create_router_metrics"]


@dataclass(frozen=True, slots=True)
class RouterMetrics:
    """The router's metric families, bound to one registry."""

    registry: CollectorRegistry
    route_decisions_total: Counter
    provider_fallbacks_total: Counter
    provider_failures_total: Counter
    spend_usd_total: Counter
    budget_denied_total: Counter
    estimated_quality: Histogram
    propensity_min: Gauge
    ope_dr_lower_bound: Gauge
    ope_ess: Gauge
    context_feature_psi: Gauge
    feedback_rejected_total: Counter
    feedback_clipped_total: Counter
    policy_activations_total: Counter
    breaker_state: Gauge


def create_router_metrics(registry: CollectorRegistry, namespace: str = "router") -> RouterMetrics:
    """Create the router metric families on ``registry``."""
    return RouterMetrics(
        registry=registry,
        route_decisions_total=Counter(
            "route_decisions_total",
            "Routing decisions by arm and policy version.",
            ["arm", "policy_version"],
            namespace=namespace,
            registry=registry,
        ),
        provider_fallbacks_total=Counter(
            "provider_fallbacks_total",
            "Decisions re-routed because an arm's circuit breaker was open.",
            ["from_arm", "to_arm"],
            namespace=namespace,
            registry=registry,
        ),
        provider_failures_total=Counter(
            "provider_failures_total",
            "Provider calls that failed, by arm and error class.",
            ["arm", "error"],
            namespace=namespace,
            registry=registry,
        ),
        spend_usd_total=Counter(
            "spend_usd_total",
            "Committed spend in USD by tenant and arm.",
            ["tenant", "arm"],
            namespace=namespace,
            registry=registry,
        ),
        budget_denied_total=Counter(
            "budget_denied_total",
            "Requests refused because no arm fitted the tenant's remaining budget.",
            ["tenant"],
            namespace=namespace,
            registry=registry,
        ),
        estimated_quality=Histogram(
            "estimated_quality",
            "Quality signal reported for served requests.",
            ["arm"],
            namespace=namespace,
            registry=registry,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ),
        propensity_min=Gauge(
            "propensity_min",
            "Smallest propensity logged since the process started.",
            namespace=namespace,
            registry=registry,
        ),
        ope_dr_lower_bound=Gauge(
            "ope_dr_lower_bound",
            "Doubly robust lower confidence bound from the most recent screening run.",
            ["candidate"],
            namespace=namespace,
            registry=registry,
        ),
        ope_ess=Gauge(
            "ope_ess",
            "Effective sample size from the most recent screening run.",
            ["candidate"],
            namespace=namespace,
            registry=registry,
        ),
        context_feature_psi=Gauge(
            "context_feature_psi",
            "Population stability index of a logged context feature against the reference window.",
            ["feature"],
            namespace=namespace,
            registry=registry,
        ),
        feedback_rejected_total=Counter(
            "feedback_rejected_total",
            "Feedback reports refused, by reason.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        feedback_clipped_total=Counter(
            "feedback_clipped_total",
            "Feedback reports whose reward was outside the allowed range and was clipped.",
            namespace=namespace,
            registry=registry,
        ),
        policy_activations_total=Counter(
            "policy_activations_total",
            "Policy activation attempts by result.",
            ["result"],
            namespace=namespace,
            registry=registry,
        ),
        breaker_state=Gauge(
            "breaker_state",
            "Circuit breaker state per arm (0 closed, 1 half-open, 2 open).",
            ["arm"],
            namespace=namespace,
            registry=registry,
        ),
    )
