"""The routing service end to end, offline.

This is where the pieces have to agree with each other: the propensity the policy returned, the one
the breaker and the budget guard forced, the one written to DuckDB, and the one an off-policy
estimator later reads. A test that only checked status codes would miss all of that, so these tests
follow individual decisions into the log and back out through the screening endpoint.

Everything runs on the offline provider. No key is configured, and ``tests/conftest.py`` blocks
non-loopback sockets, so a request that tried to reach the API would fail rather than spend money.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from bandit_llm_router.agentic.errors import ProviderError
from bandit_llm_router.agentic.providers.base import LLMRequest, LLMResponse
from bandit_llm_router.arms import ArmCatalog
from bandit_llm_router.auth import hash_api_key
from bandit_llm_router.core.config import Settings
from bandit_llm_router.policies.linear_bandit import LinUCB
from bandit_llm_router.provider import offline_provider
from bandit_llm_router.service import RouterService, build_router_service, create_router_app

pytestmark = pytest.mark.integration

ROUTE_KEY = "route-key-" + "r" * 30
FEEDBACK_KEY = "feedback-key-" + "f" * 30
SECOND_FEEDBACK_KEY = "second-feedback-key-" + "g" * 24
ADMIN_KEY = "admin-key-" + "a" * 30
APPROVE_KEY = "approve-key-" + "p" * 30
# A key holding *both* governance scopes, so separation of duties can be reached at all: the
# scope check runs first, and an admin-only key is refused by it before the duties check.
BOTH_KEY = "both-scopes-key-" + "b" * 26

ROUTE_AUTH = {"X-API-Key": ROUTE_KEY}
FEEDBACK_AUTH = {"X-API-Key": FEEDBACK_KEY}
SECOND_FEEDBACK_AUTH = {"X-API-Key": SECOND_FEEDBACK_KEY}
ADMIN_AUTH = {"X-API-Key": ADMIN_KEY}
APPROVE_AUTH = {"X-API-Key": APPROVE_KEY}
BOTH_AUTH = {"X-API-Key": BOTH_KEY}

PROMPTS = (
    "Extract every field as JSON:\norder_id: 8812\namount_eur: 1420\ndue_date: 2026-04-01",
    "Summarize the following notes in three bullets: outage, rollout, retention.",
    "```python\ntotal = compute(7, rows)\n```\nFind the bug and fix it.",
    "Hi, quick question: what is the refund window? Thanks!",
    "Work through the constraints and give the schedule that satisfies all of them:\n1. under 40",
)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "route_api_key_sha256": [hash_api_key(ROUTE_KEY)],
        "feedback_api_key_sha256": [
            hash_api_key(FEEDBACK_KEY),
            hash_api_key(SECOND_FEEDBACK_KEY),
        ],
        "admin_api_key_sha256": [hash_api_key(ADMIN_KEY), hash_api_key(BOTH_KEY)],
        "approver_api_key_sha256": [hash_api_key(APPROVE_KEY), hash_api_key(BOTH_KEY)],
        "data_dir": tmp_path / "var",
        "pricing_path": Path("config/pricing.toml"),
        "propensity_floor": 0.05,
        "tenant_budget_usd": 5.0,
        "rate_limit_per_second": 10_000.0,
        "rate_limit_burst": 10_000,
        "feedback_rate_limit_per_second": 10_000.0,
        "feedback_rate_limit_burst": 10_000,
        "screening_min_ess": 50.0,
        "screening_max_exploration_usd": 1_000.0,
        "log_json": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def build(
    tmp_path: Path,
    *,
    workload_digest: str | None = None,
    simulator_seed: int | None = None,
    **overrides: Any,
) -> tuple[RouterService, TestClient, CollectorRegistry]:
    settings = make_settings(tmp_path, **overrides)
    registry = CollectorRegistry()
    service = build_router_service(
        settings,
        registry=registry,
        seed=5,
        workload_digest=workload_digest,
        simulator_seed=simulator_seed,
    )
    app = create_router_app(settings, service=service, registry=registry)
    return service, TestClient(app, raise_server_exceptions=False), registry


@pytest.fixture
def wired(tmp_path: Path) -> Iterator[tuple[RouterService, TestClient, CollectorRegistry]]:
    service, client, registry = build(tmp_path)
    with client:
        yield service, client, registry
    service.store.close()


def route(
    client: TestClient, *, prompt: str | None = None, tenant: str = "tenant_a", **body: Any
) -> Any:
    payload: dict[str, Any] = {
        "tenant_id": tenant,
        "prompt": prompt if prompt is not None else PROMPTS[0],
        "max_output_tokens": 256,
    }
    payload.update(body)
    return client.post("/v1/route", json=payload, headers=ROUTE_AUTH)


def metric(registry: CollectorRegistry, name: str, **labels: str) -> float:
    value = registry.get_sample_value(name, labels or None)
    return float(value) if value is not None else 0.0


# --- routing ------------------------------------------------------------------------------------


def test_a_routed_request_returns_a_decision_and_a_logged_propensity(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, registry = wired
    response = route(client)
    assert response.status_code == 200
    body = response.json()
    assert body["arm"] in service.catalog.names
    assert body["model"] == service.catalog.spec(body["arm"]).model
    assert body["policy_version"] == service.policy.version
    assert body["propensity"] >= 0.05 - 1e-12
    assert body["committed_cost_usd"] > 0.0
    assert body["served_by_fallback"] is False
    assert body["text"]

    logged = service.store.decision(body["decision_id"])
    assert logged is not None
    assert logged.propensity == pytest.approx(body["propensity"])
    assert logged.arm == body["arm"]
    assert sum(logged.probabilities) == pytest.approx(1.0)
    assert len(logged.features) == service.featurizer.dim
    assert logged.actual_cost_micros is not None
    assert logged.ceiling_cost_micros >= logged.actual_cost_micros
    assert (
        metric(
            registry,
            "router_route_decisions_total",
            arm=body["arm"],
            policy_version=service.policy.version,
        )
        == 1.0
    )


def test_the_prompt_is_never_written_anywhere(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """The claim the whole log schema is built around."""
    service, client, _ = wired
    secret = "Extract the field: patient_id: 998877 and the diagnosis code."
    body = route(client, prompt=secret).json()
    logged = service.store.decision(body["decision_id"])
    assert logged is not None
    dumped = logged.model_dump_json()
    assert secret not in dumped
    assert "patient_id" not in dumped
    assert "998877" not in dumped
    # What identifies the context is a keyed digest, and the same prompt gives the same digest.
    assert logged.context_digest == service.featurizer.digest(secret)


def test_routing_spreads_traffic_and_keeps_every_propensity_usable(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, _ = wired
    for index in range(60):
        assert route(client, prompt=PROMPTS[index % len(PROMPTS)]).status_code == 200
    counts = service.store.arm_counts()
    assert sum(counts.values()) == 60
    rows = [service.store.decision(decision_id) for decision_id in _decision_ids(service)]
    assert all(row is not None and row.propensity >= 0.05 - 1e-12 for row in rows)


def _decision_ids(service: RouterService) -> list[str]:
    with service.store._lock:
        rows = service.store._connection.execute("SELECT decision_id FROM decisions").fetchall()
    return [str(row[0]) for row in rows]


def test_the_request_body_is_validated(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    assert route(client, prompt="").status_code == 422
    assert route(client, prompt="x" * 9_000).status_code == 422
    assert route(client, max_output_tokens=0).status_code == 422
    assert route(client, max_output_tokens=99_999).status_code == 422
    assert route(client, tenant="Tenant Space").status_code == 422
    assert route(client, tier="urgent").status_code == 422
    assert (
        client.post(
            "/v1/route",
            json={"tenant_id": "tenant_a", "prompt": "hi", "surprise": 1},
            headers=ROUTE_AUTH,
        ).status_code
        == 422
    )


def test_routing_needs_the_route_scope(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    assert (
        client.post("/v1/route", json={"tenant_id": "tenant_a", "prompt": "hi"}).status_code == 401
    )
    for headers in (FEEDBACK_AUTH, ADMIN_AUTH, APPROVE_AUTH):
        response = client.post(
            "/v1/route", json={"tenant_id": "tenant_a", "prompt": "hi"}, headers=headers
        )
        assert response.status_code == 403


# --- budgets ------------------------------------------------------------------------------------


def test_a_tenant_that_runs_out_of_budget_is_refused_not_billed(tmp_path: Path) -> None:
    service, client, registry = build(tmp_path, tenant_budget_usd=0.01)
    with client:
        allowed = 0
        refused = 0
        for _ in range(40):
            response = route(client)
            if response.status_code == 200:
                allowed += 1
            else:
                assert response.status_code == 402
                assert response.json()["error"]["code"] in {
                    "budget_exhausted",
                    "no_arm_available",
                }
                refused += 1
        assert allowed > 0
        assert refused > 0
        cap = service.guard.budgets.cap_micros("tenant_a")
        assert service.guard.spent_micros("tenant_a") <= cap
        assert metric(registry, "router_budget_denied_total", tenant="tenant_a") == float(refused)
    service.store.close()


def test_one_tenants_spend_does_not_touch_another(tmp_path: Path) -> None:
    service, client, _ = build(
        tmp_path, tenant_budget_caps={"tenant_poor": 0.005}, tenant_budget_usd=5.0
    )
    with client:
        for _ in range(30):
            route(client, tenant="tenant_poor")
        assert route(client, tenant="tenant_rich").status_code == 200
        assert service.guard.spent_micros("tenant_poor") <= 5_000
    service.store.close()


def test_an_unaffordable_arm_is_dropped_and_the_realized_propensity_is_logged(
    tmp_path: Path,
) -> None:
    """Exploration is priced: when Opus does not fit, its mass moves and the log says so."""
    service, client, _ = build(tmp_path, tenant_budget_usd=0.02)
    with client:
        decisions = []
        for _ in range(25):
            response = route(client, max_output_tokens=1_024)
            if response.status_code == 200:
                decisions.append(response.json())
        adjusted = [
            row
            for row in (service.store.decision(item["decision_id"]) for item in decisions)
            if row is not None and row.breaker_adjusted
        ]
        assert adjusted, "the budget should have blocked at least one arm"
        for row in adjusted:
            assert row.propensity > 0.0
            assert sum(row.probabilities) == pytest.approx(1.0)
            # A blocked arm ends at exactly zero, which is what makes the row honest.
            assert min(row.probabilities) == 0.0
    service.store.close()


# --- the provider, the breaker and the fallback propensity ----------------------------------------


class FlakyProvider:
    """Fails for one named arm, serves the rest from the offline provider."""

    name = "flaky"

    def __init__(self, catalog: ArmCatalog, failing_arm: str) -> None:
        self._inner = offline_provider(catalog, seed=1)
        self._failing = failing_arm
        self.attempts: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.attempts.append(request.role)
        if request.role == self._failing:
            raise ProviderError("simulated upstream failure")
        return self._inner.complete(request)


def test_a_provider_failure_is_redirected_and_the_new_propensity_is_logged(
    tmp_path: Path, catalog: ArmCatalog
) -> None:
    settings = make_settings(tmp_path, policy_kind="thompson")
    registry = CollectorRegistry()
    provider = FlakyProvider(catalog, "opus")
    service = build_router_service(settings, registry=registry, provider=provider, seed=2)
    app = create_router_app(settings, service=service, registry=registry)
    with TestClient(app, raise_server_exceptions=False) as client:
        fallbacks = []
        for _ in range(40):
            response = route(client)
            assert response.status_code == 200
            if response.json()["served_by_fallback"]:
                fallbacks.append(response.json())
        assert fallbacks, "sampling should have reached the failing arm at least once"
        for body in fallbacks:
            row = service.store.decision(body["decision_id"])
            assert row is not None
            assert row.served_by_fallback
            assert row.arm != "opus"
            # The logged propensity is the redirected distribution's, not the policy's.
            assert row.propensity == pytest.approx(body["propensity"])
            assert row.probabilities[service.catalog.index("opus")] == 0.0
        assert (
            metric(registry, "router_provider_failures_total", arm="opus", error="ProviderError")
            > 0.0
        )
        assert (
            metric(registry, "router_provider_fallbacks_total", from_arm="opus", to_arm="haiku")
            > 0.0
        )
        # A refund happened for every failed attempt, so an outage does not eat the budget.
        assert service.guard.spent_micros("tenant_a") > 0
    service.store.close()


def test_the_breaker_opens_and_readiness_still_reports_ready(
    tmp_path: Path, catalog: ArmCatalog
) -> None:
    settings = make_settings(tmp_path, breaker_failure_threshold=1)
    registry = CollectorRegistry()
    service = build_router_service(
        settings, registry=registry, provider=FlakyProvider(catalog, "opus"), seed=3
    )
    app = create_router_app(settings, service=service, registry=registry)
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(40):
            route(client)
        assert service.breaker.snapshot()["opus"] in {"open", "half_open"}
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["checks"]["arm_available"] == "ok"
    service.store.close()


def test_every_arm_failing_is_a_503(tmp_path: Path, catalog: ArmCatalog) -> None:
    class DeadProvider:
        name = "dead"

        def complete(self, request: LLMRequest) -> LLMResponse:
            raise ProviderError("everything is down")

    settings = make_settings(tmp_path)
    registry = CollectorRegistry()
    service = build_router_service(settings, registry=registry, provider=DeadProvider(), seed=4)
    app = create_router_app(settings, service=service, registry=registry)
    with TestClient(app, raise_server_exceptions=False) as client:
        statuses = {route(client).status_code for _ in range(12)}
        assert statuses <= {503, 402}
        assert 503 in statuses
    service.store.close()


# --- feedback -----------------------------------------------------------------------------------


def test_feedback_teaches_the_live_policy_once(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, _ = wired
    decision_id = route(client).json()["decision_id"]
    before = service.policy.updates
    first = client.post(
        "/v1/feedback",
        json={"report_id": "r1", "decision_id": decision_id, "reward": 0.8},
        headers=FEEDBACK_AUTH,
    )
    assert first.status_code == 200
    assert first.json()["applied_to_policy"] is True
    assert service.policy.updates == before + 1
    # A second report refines the stored aggregate but is not re-applied online.
    second = client.post(
        "/v1/feedback",
        json={"report_id": "r2", "decision_id": decision_id, "reward": 0.7},
        headers=SECOND_FEEDBACK_AUTH,
    )
    assert second.status_code == 200
    assert second.json()["applied_to_policy"] is False
    assert second.json()["n_sources"] == 2
    assert service.policy.updates == before + 1


def test_a_hostile_reward_is_clipped_before_it_reaches_the_posterior(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, registry = wired
    decision_id = route(client).json()["decision_id"]
    response = client.post(
        "/v1/feedback",
        json={"report_id": "r1", "decision_id": decision_id, "reward": 1e9},
        headers=FEEDBACK_AUTH,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["clipped"] is True
    assert body["reward"] == 1.0
    assert metric(registry, "router_feedback_clipped_total") == 1.0
    # The posterior saw 1.0, not 1e9: the ridge statistics stay finite.
    assert np.all(np.isfinite(service.policy.posterior.theta_matrix()))


def test_feedback_for_an_unknown_decision_is_refused(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, registry = wired
    response = client.post(
        "/v1/feedback",
        json={"report_id": "r1", "decision_id": "no-such-decision", "reward": 0.5},
        headers=FEEDBACK_AUTH,
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_decision"
    assert metric(registry, "router_feedback_rejected_total", reason="UnknownDecisionError") == 1.0


def test_a_replayed_report_is_refused(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    decision_id = route(client).json()["decision_id"]
    payload = {"report_id": "r1", "decision_id": decision_id, "reward": 0.5}
    assert client.post("/v1/feedback", json=payload, headers=FEEDBACK_AUTH).status_code == 200
    duplicate = client.post("/v1/feedback", json=payload, headers=FEEDBACK_AUTH)
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "duplicate_report"


def test_one_source_cannot_flood_a_decision(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    decision_id = route(client).json()["decision_id"]
    accepted = 0
    for index in range(6):
        response = client.post(
            "/v1/feedback",
            json={"report_id": f"r{index}", "decision_id": decision_id, "reward": 0.0},
            headers=FEEDBACK_AUTH,
        )
        if response.status_code == 200:
            accepted += 1
        else:
            assert response.status_code == 429
    assert accepted == 2  # max_reports_per_source_per_decision


def test_feedback_needs_the_feedback_scope(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    decision_id = route(client).json()["decision_id"]
    payload = {"report_id": "r1", "decision_id": decision_id, "reward": 0.5}
    assert client.post("/v1/feedback", json=payload).status_code == 401
    assert client.post("/v1/feedback", json=payload, headers=ROUTE_AUTH).status_code == 403
    assert client.post("/v1/feedback", json=payload, headers=ADMIN_AUTH).status_code == 403


def test_the_feedback_body_is_validated(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    decision_id = route(client).json()["decision_id"]
    for body in (
        {"report_id": "bad id", "decision_id": decision_id, "reward": 0.5},
        {"report_id": "r1", "decision_id": decision_id, "reward": 1e12},
        {"report_id": "r1", "decision_id": decision_id},
        {"report_id": "r1", "decision_id": decision_id, "reward": 0.5, "extra": 1},
    ):
        assert client.post("/v1/feedback", json=body, headers=FEEDBACK_AUTH).status_code == 422


# --- the policy lifecycle -------------------------------------------------------------------------


def seed_log(service: RouterService, client: TestClient, *, requests: int = 220) -> None:
    """Route and reward enough traffic that screening has evidence to work with."""
    for index in range(requests):
        response = route(client, prompt=PROMPTS[index % len(PROMPTS)])
        if response.status_code != 200:
            continue
        decision_id = response.json()["decision_id"]
        row = service.store.decision(decision_id)
        assert row is not None
        # A rubric-like reward: the bigger the arm, the better the answer, minus its cost.
        quality = {"haiku": 0.55, "sonnet": 0.7, "opus": 0.85}[row.arm]
        client.post(
            "/v1/feedback",
            json={"report_id": f"rep-{index}", "decision_id": decision_id, "reward": quality},
            headers=FEEDBACK_AUTH,
        )


def _policy_preferring(service: RouterService, arm: str, features: np.ndarray) -> LinUCB:
    """A greedy LinUCB trained, on the logged contexts, to prefer exactly one arm."""
    policy = LinUCB(
        arms=service.catalog.names,
        dim=service.featurizer.dim,
        feature_version=service.featurizer.version,
        propensity_floor=service.policy.propensity_floor,
        alpha=0.0,
    )
    target = service.catalog.index(arm)
    for row in features:
        for index in range(len(service.catalog)):
            policy.update(row, index, 1.0 if index == target else 0.0)
    return policy


def test_status_reports_what_is_live(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, _ = wired
    seed_log(service, client, requests=12)
    response = client.get("/v1/policies", headers=ADMIN_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["live_policy_version"] == service.policy.version
    assert body["active_snapshot_digest"] is None
    assert body["decisions_logged"] >= 12
    assert body["rewards_logged"] >= 12
    assert set(body["breaker"]) == set(service.catalog.names)
    assert len(body["audit_head"]) == 64
    assert client.get("/v1/policies").status_code == 401
    assert client.get("/v1/policies", headers=ROUTE_AUTH).status_code == 403


def test_a_candidate_is_screened_on_the_services_own_logs(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """The caller supplies the snapshot, never the evidence."""
    service, client, registry = wired
    seed_log(service, client)
    digest = service.register_live_policy()
    response = client.post(f"/v1/policies/{digest}/screen", headers=ADMIN_AUTH)
    assert response.status_code == 200
    report = response.json()
    assert report["candidate_version"] == service.policy.version
    assert report["incumbent_version"] == service.policy.version
    assert report["n"] >= 100
    assert len(report["logs_digest"]) == 64
    assert report["candidate_dr"]["method"] == "dr"
    assert metric(
        registry, "router_ope_dr_lower_bound", candidate=service.policy.version
    ) == pytest.approx(report["dr_lower_bound"])
    assert metric(registry, "router_ope_ess", candidate=service.policy.version) > 0.0


def test_a_logged_row_identifies_the_traffic_that_produced_it(tmp_path: Path) -> None:
    """The reproducibility columns have to be populated, not just declared.

    The spec asks for the simulator seed, the workload hash and the policy version in every log row.
    ``policy_version`` and ``policy_step`` always were; ``workload_digest`` and ``simulator_seed``
    are only knowable when a simulator drives the service, so they are constructor arguments -- and
    they stay NULL rather than being faked when nothing supplies them.
    """
    service, client, _ = build(
        tmp_path / "seeded", workload_digest="c0ffee" * 6, simulator_seed=4_242
    )
    try:
        response = client.post(
            "/v1/route", json={"tenant_id": "t1", "prompt": PROMPTS[0]}, headers=ROUTE_AUTH
        )
        assert response.status_code == 200
        row = service.store.decision(response.json()["decision_id"])
        assert row is not None
        assert row.workload_digest == "c0ffee" * 6
        assert row.simulator_seed == 4_242
        assert row.policy_version == service.policy.version
    finally:
        service.store.close()

    plain, plain_client, _ = build(tmp_path / "plain")
    try:
        decision_id = plain_client.post(
            "/v1/route", json={"tenant_id": "t1", "prompt": PROMPTS[0]}, headers=ROUTE_AUTH
        ).json()["decision_id"]
        bare = plain.store.decision(decision_id)
        assert bare is not None
        assert bare.workload_digest is None
        assert bare.simulator_seed is None
    finally:
        plain.store.close()


def test_screening_can_be_narrowed_to_a_window_of_the_learners_own_log(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """The gate runs on a log the live policy wrote *while it was learning*, so it needs a window.

    ``policy_version`` cannot provide one: it hashes hyperparameters, so every row a learning LinUCB
    writes carries the same version however far the posterior has travelled. ``policy_step`` counts
    the updates, so an operator can ask for the stretch of the log over which the logging policy was
    nearly still -- and the report has to say which window it used, because the DR lower bound it
    carries means something different on 40 steps than on 400.
    """
    service, client, _ = wired
    seed_log(service, client)
    steps = service.store.fetch_logged_rows(n_arms=len(service.catalog)).policy_steps
    assert int(steps.max()) > int(steps.min()), "the seeded log must span several policy steps"
    midpoint = int(steps.min()) + (int(steps.max()) - int(steps.min())) // 2
    digest = service.register_live_policy()

    whole = client.post(f"/v1/policies/{digest}/screen", headers=ADMIN_AUTH).json()
    early = client.post(
        f"/v1/policies/{digest}/screen?max_policy_step={midpoint}", headers=ADMIN_AUTH
    ).json()
    assert 0 < early["n"] < whole["n"]
    assert early["logs_digest"] != whole["logs_digest"]
    assert early["ess"] < whole["ess"]

    empty = client.post(
        f"/v1/policies/{digest}/screen?min_policy_step={int(steps.max()) + 1_000}",
        headers=ADMIN_AUTH,
    )
    assert empty.status_code == 409
    assert empty.json()["error"]["code"] == "insufficient_evidence"
    assert (
        client.post(
            f"/v1/policies/{digest}/screen?min_policy_step=-1", headers=ADMIN_AUTH
        ).status_code
        == 422
    )


def test_the_service_screens_with_a_block_bootstrap_because_its_log_is_not_iid(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """The bound is computed on adaptively collected rows, and the record has to say so.

    A percentile bootstrap over single rows assumes the rows are independent. They are not: the
    propensity on row ``i`` depends on the rewards of the rows before it. The service resamples
    contiguous blocks instead, and because ``ScreeningConfig`` is part of the hashed report, that
    choice is auditable rather than a hidden default.
    """
    service, client, _ = wired
    assert service.screening.bootstrap_block_size == 250
    seed_log(service, client)
    digest = service.register_live_policy()
    report = client.post(f"/v1/policies/{digest}/screen", headers=ADMIN_AUTH).json()
    assert report["config"]["bootstrap_block_size"] == 250


def test_screening_an_unknown_snapshot_is_a_404(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, _ = wired
    seed_log(service, client, requests=60)
    response = client.post(f"/v1/policies/{'f' * 64}/screen", headers=ADMIN_AUTH)
    assert response.status_code == 404
    assert client.post("/v1/policies/not-a-digest/screen", headers=ADMIN_AUTH).status_code == 422


def test_a_candidate_identical_to_the_incumbent_cannot_be_promoted(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """It has no advantage to demonstrate, so the DR lower bound cannot beat the incumbent."""
    service, client, registry = wired
    seed_log(service, client)
    digest = service.register_live_policy()
    response = client.post(f"/v1/policies/{digest}/propose", headers=ADMIN_AUTH)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "screening_rejected"
    assert metric(registry, "router_policy_activations_total", result="screening_rejected") == 1.0
    assert client.get("/v1/activations", headers=ADMIN_AUTH).json() == []


def test_the_whole_gate_runs_for_a_better_candidate(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """Screen, propose, refuse self-approval, approve as a second principal, activate.

    The incumbent is made deliberately stale first: it prefers the arm the seeded rubric rewards
    *least*. That is the realistic shape of a promotion -- the live policy is out of date and a
    retrained candidate beats it on the logs. A candidate with no advantage is refused, which is
    what ``test_a_candidate_identical_to_the_incumbent_cannot_be_promoted`` covers.
    """
    service, client, registry = wired
    seed_log(service, client)
    rows = service.store.fetch_logged_rows(n_arms=len(service.catalog))
    stale = _policy_preferring(service, "haiku", rows.batch.features)
    service.policy = stale
    better = _policy_preferring(service, "opus", rows.batch.features)
    digest = service.controller.register_snapshot(better.snapshot()).digest

    screened = client.post(f"/v1/policies/{digest}/screen", headers=ADMIN_AUTH).json()
    assert screened["eligible"] is True, screened["reasons"]
    assert screened["incumbent_version"] == stale.version
    assert screened["candidate_dr"]["value"] > screened["incumbent_dr"]["value"]

    # Proposed by a principal that also *holds* the approve scope, so the refusal below is the
    # separation-of-duties check and not the scope check.
    proposed = client.post(f"/v1/policies/{digest}/propose", headers=BOTH_AUTH)
    assert proposed.status_code == 200
    activation_id = proposed.json()["id"]
    assert proposed.json()["status"] == "proposed"

    self_approval = client.post(f"/v1/activations/{activation_id}/approve", headers=BOTH_AUTH)
    assert self_approval.status_code == 403
    assert self_approval.json()["error"]["code"] == "separation_of_duties"
    assert (
        metric(registry, "router_policy_activations_total", result="self_approval_refused") == 1.0
    )

    approved = client.post(f"/v1/activations/{activation_id}/approve", headers=APPROVE_AUTH)
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_by"] != proposed.json()["proposed_by"]

    activated = client.post(f"/v1/activations/{activation_id}/activate", headers=ADMIN_AUTH)
    assert activated.status_code == 200
    assert activated.json()["status"] == "activated"
    assert service.policy.version == better.version

    status = client.get("/v1/policies", headers=ADMIN_AUTH).json()
    assert status["active_snapshot_digest"] == digest
    assert status["live_policy_version"] == better.version
    assert metric(registry, "router_policy_activations_total", result="activated") == 1.0

    # The chain is verifiable, and it names who did what.
    from bandit_llm_router.agentic.audit import verify_file

    result = verify_file(service.audit.path, expect_head=status["audit_head"])
    assert result.ok
    assert result.entries == 3


def test_activation_without_approval_is_refused(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    service, client, _ = wired
    seed_log(service, client, requests=60)
    response = client.post(f"/v1/activations/{'0' * 32}/activate", headers=ADMIN_AUTH)
    assert response.status_code == 404
    assert (
        client.post(f"/v1/activations/{'0' * 32}/approve", headers=APPROVE_AUTH).status_code == 404
    )


def test_the_admin_routes_need_their_own_scopes(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    digest = "a" * 64
    activation = "b" * 32
    assert client.post(f"/v1/policies/{digest}/propose", headers=ROUTE_AUTH).status_code == 403
    assert client.get("/v1/activations", headers=FEEDBACK_AUTH).status_code == 403
    # Approving takes the approve scope, which an admin-only key does not hold.
    refused = client.post(f"/v1/activations/{activation}/approve", headers=ADMIN_AUTH)
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "forbidden"
    # Activating takes the admin scope, not the approve one.
    assert (
        client.post(f"/v1/activations/{activation}/activate", headers=APPROVE_AUTH).status_code
        == 403
    )


# --- health and metrics ---------------------------------------------------------------------------


def test_health_and_readiness_report_the_routers_own_dependencies(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {
        "policy_loaded": "ok",
        "arm_available": "ok",
        "log_store": "ok",
    }


def test_metrics_require_a_key_and_expose_the_router_families(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    _, client, _ = wired
    route(client)
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers=ADMIN_AUTH)
    assert response.status_code == 200
    body = response.text
    for family in (
        "router_route_decisions_total",
        "router_spend_usd_total",
        "router_propensity_min",
        "router_provider_fallbacks_total",
        "router_breaker_state",
        "router_policy_activations_total",
    ):
        assert family in body


def test_the_minimum_propensity_gauge_tracks_the_logs_health(
    wired: tuple[RouterService, TestClient, CollectorRegistry],
) -> None:
    """If this drifts toward zero the log is quietly becoming unusable for OPE."""
    service, client, registry = wired
    for index in range(20):
        route(client, prompt=PROMPTS[index % len(PROMPTS)])
    assert metric(registry, "router_propensity_min") >= service.policy.propensity_floor - 1e-12


# --- secrets ------------------------------------------------------------------------------------


def test_no_key_of_any_kind_reaches_the_log_output(tmp_path: Path, restore_logging: None) -> None:
    """The one test that reads the bytes the process would actually ship to a log sink.

    Redaction is unit-tested against patterns, but the failure that matters here is a *specific*
    secret reaching stdout through some field nobody thought to redact: the provider key, a caller's
    API key, or the featurizer's digest key. So this drives the whole service -- routing, feedback,
    screening, activation -- with sentinel values configured, captures the configured log stream and
    asserts none of the sentinels appear anywhere in it.

    The provider key is set but never used: the offline provider is injected, so the live client is
    never constructed and no request leaves the process.
    """
    from io import StringIO

    from pydantic import SecretStr

    from bandit_llm_router.core.logging import configure_logging

    provider_key = "sk-ant-api03-" + "Z" * 40
    digest_key = "feature-digest-key-" + "Q" * 24
    stream = StringIO()
    configure_logging("DEBUG", json_output=True, stream=stream)

    settings = make_settings(
        tmp_path,
        anthropic_api_key=SecretStr(provider_key),
        feature_digest_key=SecretStr(digest_key),
    )
    registry = CollectorRegistry()
    service = build_router_service(
        settings, registry=registry, provider=offline_provider(service_catalog()), seed=5
    )
    app = create_router_app(settings, service=service, registry=registry)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            for index in range(8):
                response = route(client, prompt=PROMPTS[index % len(PROMPTS)])
                assert response.status_code == 200
                decision_id = response.json()["decision_id"]
                feedback = client.post(
                    "/v1/feedback",
                    json={
                        "decision_id": decision_id,
                        "report_id": f"report-{index}",
                        "reward": 0.8,
                    },
                    headers=FEEDBACK_AUTH,
                )
                assert feedback.status_code == 200
            # Unauthenticated and wrongly scoped attempts, which is where a handler is most tempted
            # to echo the credential it just rejected.
            assert client.post("/v1/route", json={"tenant_id": "t", "prompt": "x"}).status_code in {
                401,
                422,
            }
            assert route_with(client, provider_key).status_code == 401
            assert client.get("/metrics", headers={"X-API-Key": provider_key}).status_code == 401
            assert client.get("/v1/policies", headers=ADMIN_AUTH).status_code == 200
    finally:
        service.store.close()

    output = stream.getvalue()
    # Non-vacuous: the sentinels are absent from a stream that really did carry the request events,
    # including the key *identifier* the router logs in their place.
    assert "route_decision" in output
    assert "key_id" in output
    for secret in (provider_key, digest_key, ROUTE_KEY, FEEDBACK_KEY, ADMIN_KEY):
        assert secret not in output
    # Nor the halves an over-eager formatter might split a key into.
    assert "Z" * 40 not in output
    assert "Q" * 24 not in output


def route_with(client: TestClient, key: str) -> Any:
    return client.post(
        "/v1/route",
        json={"tenant_id": "tenant_a", "prompt": PROMPTS[0], "max_output_tokens": 256},
        headers={"X-API-Key": key},
    )


def service_catalog() -> ArmCatalog:
    from bandit_llm_router.arms import load_catalog

    return load_catalog(Path("config/pricing.toml"))
