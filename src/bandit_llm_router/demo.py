"""The offline demo: one simulated day, end to end, with no API key and no network.

``uv run python tasks.py demo`` runs this. It is the shortest honest tour of the repo:

1. **A day of traffic.** LinUCB, Thompson sampling, epsilon-greedy, a hand-written keyword rule and
   the three always-one-model baselines are run over the *same* request stream, and the table
   reports reward, quality, cost per thousand requests, the saving against always-Sonnet, and regret
   against both the unconstrained oracle and the clairvoyant policy with the same propensity floor.
2. **A mid-day quality shift.** One model quietly gets worse at code. Stationary and discounted
   posteriors are compared on how many requests they need to get back to 95% of oracle reward.
3. **Off-policy evaluation.** A frozen policy logs decisions; IPS, SNIPS and DR then estimate the
   value of candidates the log never ran, next to the exact value the simulator can compute.
4. **Screening and activation.** Two sweeps: trained candidates against the cost-blind status quo
   (admitted, proposed, approved by a second principal, activated) and static arms against a
   converged incumbent (refused). Every verdict is printed next to the candidate's *exact* policy
   value, so the gate can be checked for being right rather than merely for firing. The activation
   audit chain is verified at the end.
5. **Hostile feedback.** The same poisoned reward stream is replayed four ways: with no defenses,
   with clipping only, with the full aggregation that protects the *stored* reward, and through the
   channel the *online posterior* really gets (one clipped report at the reduced single-source
   weight). The last two are different numbers and the table says so.
6. **The live path, offline.** Real requests go through ``RouterService`` with the offline provider:
   DuckDB logging, per-tenant budget caps (including a tenant that runs out), feedback ingestion and
   the policy learning from it.
7. **Tracking.** The run is logged to MLflow's local file backend.

Everything is seeded, so two runs print the same numbers.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from bandit_llm_router.activation import POLICY_MODEL_NAME
from bandit_llm_router.agentic.audit import verify_file
from bandit_llm_router.arms import load_catalog
from bandit_llm_router.auth import Scope, ScopedApiKeyVerifier, ScopedPrincipal, hash_api_key
from bandit_llm_router.core.config import Settings
from bandit_llm_router.experiments import (
    AggregationMode,
    PoisonedFeedbackChannel,
    expected_policy_cost,
    run_ope_accuracy,
    run_shift_experiment,
    run_simulation,
    true_policy_value,
)
from bandit_llm_router.feedback import FeedbackSubmission
from bandit_llm_router.ope.screening import ScreeningConfig
from bandit_llm_router.policies import (
    EpsilonGreedy,
    KeywordRule,
    LinearThompson,
    LinUCB,
    StaticArm,
)
from bandit_llm_router.policies.linear_bandit import LinearBanditPolicy
from bandit_llm_router.reporting import markdown_table
from bandit_llm_router.service import RouteRequest, build_router_service
from bandit_llm_router.simulator import QualityShift, SimulatedEnvironment, Workload
from bandit_llm_router.simulator.workload import FAMILIES
from bandit_llm_router.tracking import log_policy_run

__all__ = ["run_demo"]

_REQUESTS = 25_000
_LOG_REQUESTS = 12_000
_WARMUP = 3_000
_POISON_REQUESTS = 8_000
_FLOOR = 0.02
_SEED = 7
_BOOTSTRAP = 200
# The screening cost cap for the demo's horizon of 100,000 requests. See ``_screening_section``.
_MAX_EXPLORATION_USD = 300.0


def _make_policy(kind: str, environment: SimulatedEnvironment, **overrides: Any) -> Any:
    arms, dim = environment.arms, environment.feature_dim
    version = environment.featurizer.version
    common: dict[str, Any] = {
        "arms": arms,
        "dim": dim,
        "feature_version": version,
        "propensity_floor": _FLOOR,
    }
    if kind == "linucb":
        return LinUCB(**{"alpha": 0.5, **common, **overrides})
    if kind == "thompson":
        return LinearThompson(**{"posterior_sd": 0.3, "mc_samples": 24, **common, **overrides})
    if kind == "epsilon_greedy":
        return EpsilonGreedy(**{"epsilon": 0.05, **common, **overrides})
    if kind == "keyword_rule":
        return KeywordRule(
            arms=arms,
            dim=dim,
            feature_version=version,
            code_arm="opus",
            long_arm="sonnet",
            default_arm="haiku",
            propensity_floor=_FLOOR,
        )
    return StaticArm(
        arms=arms, dim=dim, feature_version=version, target=kind, propensity_floor=_FLOOR
    )


def _learning_section(
    environment: SimulatedEnvironment,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stream = environment.workload.sample_indices(_REQUESTS, np.random.default_rng(99))
    kinds = ("linucb", "thompson", "epsilon_greedy", "keyword_rule", "haiku", "sonnet", "opus")
    results = {}
    for kind in kinds:
        policy = _make_policy(kind, environment)
        learn = kind in {"linucb", "thompson", "epsilon_greedy"}
        results[kind] = run_simulation(
            environment, policy, n=_REQUESTS, seed=_SEED, learn=learn, stream=stream
        )
    sonnet = results["sonnet"]
    rows = []
    for kind, result in results.items():
        rows.append(
            {
                "policy": kind,
                "mean_reward": round(result.mean_reward, 4),
                "mean_quality": round(result.mean_quality, 4),
                "usd_per_1k_requests": round(result.mean_cost_micros / 1_000, 3),
                "saving_vs_sonnet": round(1 - result.mean_cost_micros / sonnet.mean_cost_micros, 3),
                "quality_delta_vs_sonnet": round(result.mean_quality - sonnet.mean_quality, 4),
                "oracle_regret": round(result.final_regret, 1),
                "in_class_regret": round(result.final_class_regret, 1),
                "in_class_slope": round(result.class_regret_slope, 3),
                "arm_shares": tuple(round(share, 3) for share in result.arm_shares),
            }
        )
    best = results["linucb"]
    summary = {
        "requests": _REQUESTS,
        "linucb_saving_vs_sonnet": round(1 - best.mean_cost_micros / sonnet.mean_cost_micros, 4),
        "linucb_quality_delta_vs_sonnet": round(best.mean_quality - sonnet.mean_quality, 4),
        "linucb_in_class_regret": round(best.final_class_regret, 1),
    }
    return rows, summary


def _shift_section(catalog: Any, workload: Workload) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A mid-day model update that degrades the arm the router had learned to prefer.

    Degrading one arm on one prompt family (the first thing one reaches for) is realistic but too
    small to see: it moves a few percent of traffic and both posteriors shrug it off. Haiku carries
    roughly three quarters of the decisions here, so degrading *it* across every family is what
    makes "requests to recover" a number rather than a zero.
    """
    shift_step = _REQUESTS // 2
    shifted = SimulatedEnvironment(
        workload=workload,
        catalog=catalog,
        shifts=tuple(
            QualityShift(at_step=shift_step, family=family, arm="haiku", intercept_delta=-3.5)
            for family in FAMILIES
        ),
    )
    policies = {
        "linucb_stationary": (1.0, _make_policy("linucb", shifted, discount=1.0)),
        "linucb_discounted": (0.9995, _make_policy("linucb", shifted, discount=0.9995)),
    }
    recoveries = run_shift_experiment(
        shifted,
        policies=policies,
        n=_REQUESTS,
        seed=_SEED,
        shift_step=shift_step,
        window=2_000,
    )
    rows = [recovery.summary_row() for recovery in recoveries]
    summary = {
        item.policy_name: item.recovery_requests
        if item.recovery_requests is not None
        else "not recovered"
        for item in recoveries
    }
    return rows, summary


def _ope_section(
    environment: SimulatedEnvironment,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        "always_haiku": lambda env, seed: _make_policy("haiku", env),
        "always_opus": lambda env, seed: _make_policy("opus", env),
        "keyword_rule": lambda env, seed: _make_policy("keyword_rule", env),
        "trained_thompson": lambda env, seed: _trained(env, "thompson", seed),
    }
    accuracy, details = run_ope_accuracy(
        environment,
        logger_factory=lambda: _make_policy("linucb", environment),
        candidates=candidates,
        n_logs=_LOG_REQUESTS,
        seeds=(_SEED,),
        warmup=_WARMUP,
        bootstrap=_BOOTSTRAP,
    )
    accuracy_rows = [accuracy[method].summary_row() for method in ("ips", "snips", "dr")]
    detail_rows = [row for row in details if row["estimator"] == "dr"]
    return accuracy_rows, detail_rows


def _trained(
    environment: SimulatedEnvironment, kind: str, seed: int, *, steps: int | None = None
) -> LinearBanditPolicy:
    policy: LinearBanditPolicy = _make_policy(kind, environment)
    run_simulation(
        environment, policy, n=steps if steps is not None else _WARMUP, seed=seed + 5, learn=True
    )
    return policy


def _incumbent_converged(environment: SimulatedEnvironment) -> LinearBanditPolicy:
    """The other realistic incumbent: a LinUCB that has already learned this workload.

    Sweep 2 needs this. Against the "always the largest model" status quo *nothing* in this
    environment is genuinely worse -- the cost-penalized reward makes pure Opus the worst
    deterministic policy there is -- so a refusal in that sweep could only ever be a candidate
    identical to the incumbent, which proves nothing about the gate. A converged incumbent is the
    setting where "genuinely worse candidate" has instances.
    """
    return _trained(environment, "linucb", _SEED + 11, steps=_WARMUP * 3)


def _incumbent_always_largest(environment: SimulatedEnvironment) -> LinearBanditPolicy:
    """The status quo this repo exists to replace: send everything to the largest model.

    It is a real :class:`LinUCB` with real posteriors -- taught, on the pool's own contexts, that
    the dearest arm always scores and the others never do -- so the screening comparison is between
    two policies of the same class, not between a policy and a hand-written stub. Its exploration is
    switched off (``alpha=0``) because a deployed "always use the big model" rule does not explore.

    Under the cost-penalized reward this incumbent is genuinely poor (its true value is around 0.41
    against roughly 0.65 for a trained policy), which is the point: paying Opus prices for
    small-talk and extraction buys quality nobody asked for.
    """
    policy: LinearBanditPolicy = _make_policy("linucb", environment, alpha=0.0)
    features = environment.feature_matrix
    dearest = int(np.argmax(environment.cost_matrix.mean(axis=0)))
    for row in features:
        for arm in range(environment.n_arms):
            policy.update(row, arm, 1.0 if arm == dearest else 0.0)
    return policy


def _screening_section(
    environment: SimulatedEnvironment, workdir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two sweeps, because one incumbent cannot exercise both sides of the gate.

    The logging policy is a lightly trained LinUCB in both sweeps, so the *evidence* is identical
    and only the incumbent and the candidates change.

    * **Against the stale cost-blind incumbent** -- the README's status quo, everything to the
      largest model -- two freshly trained policies are genuinely better, and the gate admits them.
      One of them is then proposed, approved by a second principal and activated for real.
    * **Against a converged incumbent** -- a LinUCB that already learned this workload -- the static
      arms are genuinely worse, and the gate refuses them.

    Why two sweeps rather than one: under the cost-penalized reward, pure Opus is the *worst*
    deterministic policy in this environment, so against the always-largest incumbent no candidate
    can be genuinely worse. A refusal in that sweep could only be a candidate identical to the
    incumbent -- whose own DR lower bound can never beat its own point estimate, making the refusal
    arithmetically certain and evidentially empty. Splitting the sweeps is what makes both verdicts
    mean something, and every one of them is checked against ``true_value`` below.
    """
    from bandit_llm_router.ope.screening import screen_candidate  # noqa: PLC0415

    logger = _trained(environment, "linucb", _SEED)
    logging_run = run_simulation(environment, logger, n=_LOG_REQUESTS, seed=_SEED + 1, learn=False)
    batch = logging_run.log.to_batch(environment)
    expected = environment.cost_matrix[logging_run.log.pool_indices]
    ceiling = environment.ceiling_matrix[logging_run.log.pool_indices]
    config = ScreeningConfig(
        min_ess=200.0,
        bootstrap=_BOOTSTRAP,
        horizon_requests=100_000,
        # Sized between the candidates on purpose. Measured on this workload at a horizon of
        # 100,000 requests: a trained, cost-aware policy's bound lands near 220 USD and
        # ``always_sonnet`` near 270, so neither is refused on cost; ``always_opus``, which parks
        # 96% of its mass on the dearest arm, lands near 650 and is. So the cost gate is the *only*
        # difference between the two refusals, which is what makes it a gate rather than a constant.
        # (At full size the static arms' ESS is ~300 against the 200 minimum, so the evidence gate
        # does not fire here; it does at the reduced sizes the integration test runs, where a few
        # thousand rows really is thin evidence for a policy concentrated on one rarely logged arm.)
        max_exploration_usd=_MAX_EXPLORATION_USD,
    )
    sweeps: tuple[tuple[str, LinearBanditPolicy, dict[str, Any]], ...] = (
        (
            "stale cost-blind",
            _incumbent_always_largest(environment),
            {
                "trained_thompson": _trained(environment, "thompson", _SEED + 2),
                "trained_linucb": _trained(environment, "linucb", _SEED + 6),
            },
        ),
        (
            "converged",
            _incumbent_converged(environment),
            {
                "always_opus": _make_policy("opus", environment),
                "always_sonnet": _make_policy("sonnet", environment),
            },
        ),
    )

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    incumbent_truths: dict[str, float] = {}
    for label, incumbent, sweep_candidates in sweeps:
        incumbent_truths[label] = round(
            true_policy_value(
                environment,
                incumbent,
                logging_run.log.pool_indices,
                rng=np.random.default_rng(_SEED + 8),
            ),
            4,
        )
        for name, candidate in sweep_candidates.items():
            report = screen_candidate(
                batch,
                candidate=candidate,
                incumbent=incumbent,
                expected_cost_micros=expected,
                ceiling_cost_micros=ceiling,
                config=config,
                rng=np.random.default_rng(_SEED + 3),
            )
            reports[name] = report
            candidates[name] = candidate
            truth = round(
                true_policy_value(
                    environment,
                    candidate,
                    logging_run.log.pool_indices,
                    rng=np.random.default_rng(_SEED + 4),
                ),
                4,
            )
            rows.append(
                {
                    **report.summary_row(),
                    "candidate": name,
                    "incumbent": label,
                    "true_value": truth,
                    "incumbent_true_value": incumbent_truths[label],
                }
            )
            # Kept so a test can check the gate was *right*, not merely that it fired.
            verdicts[name] = {
                "eligible": report.eligible,
                "reasons": list(report.reasons),
                "true_value": truth,
                "incumbent": label,
                "incumbent_true_value": incumbent_truths[label],
                "horizon_usd": round(report.cost.upper_bound_horizon_usd, 2),
            }

    # Register, propose, approve and activate the eligible candidate through the real controller.
    settings = Settings(_env_file=None, data_dir=workdir / "service", environment="test")
    service = build_router_service(settings, seed=_SEED)
    eligible = next((name for name, report in reports.items() if report.eligible), None)
    activation: dict[str, Any] = {"activated": False}
    if eligible is not None:
        digest = service.controller.register_snapshot(candidates[eligible].snapshot()).digest
        record = service.controller.propose(
            snapshot_digest=digest, report=reports[eligible], proposed_by="operator_a"
        )
        service.controller.approve(record.id, approver="operator_b")
        activated = service.controller.activate(record.id, actor="operator_a")
        activation = {
            "activated": True,
            "candidate": eligible,
            "activation_id": activated.id,
            "snapshot_digest": digest[:16],
            "audit_head": service.audit.head[:16],
        }
    refused = {}
    for name, report in reports.items():
        if report.eligible:
            continue
        snapshot_digest = service.controller.register_snapshot(candidates[name].snapshot()).digest
        try:
            service.controller.propose(
                snapshot_digest=snapshot_digest, report=report, proposed_by="operator_a"
            )
        except Exception as exc:  # noqa: BLE001 - the demo shows the refusal, not the traceback
            refused[name] = type(exc).__name__
    verification = verify_file(service.audit.path)
    service.store.close()
    return rows, {
        **activation,
        "refused": refused,
        "audit_ok": verification.ok,
        "audit_entries": verification.entries,
        "incumbent_true_values": incumbent_truths,
        "verdicts": verdicts,
    }


def _poisoning_section(environment: SimulatedEnvironment) -> list[dict[str, Any]]:
    stream = environment.workload.sample_indices(_POISON_REQUESTS, np.random.default_rng(31))
    rows = []
    clean = run_simulation(
        environment,
        _make_policy("linucb", environment),
        n=_POISON_REQUESTS,
        seed=_SEED,
        stream=stream,
    )
    rows.append(
        {
            "feedback": "honest only",
            "defenses": "n/a",
            "mean_decision_value": round(float(clean.log.decision_values.mean()), 4),
            "value_drop": 0.0,
        }
    )
    baseline = float(clean.log.decision_values.mean())
    # The last row is the one the deployed online learner actually gets: the service teaches the
    # posterior from the *first* accepted report for a decision, so the per-source median and the
    # weighted median never run there. The row above it is the stored reward that OPE reads.
    ablation: tuple[tuple[AggregationMode, str], ...] = (
        ("none", "none"),
        ("clip_only", "clipping only"),
        ("robust", "full aggregation (stored reward)"),
        ("first_report_only", "online posterior (first report, clipped)"),
    )
    for mode, label in ablation:
        result = run_simulation(
            environment,
            _make_policy("linucb", environment),
            n=_POISON_REQUESTS,
            seed=_SEED,
            stream=stream,
            channel=PoisonedFeedbackChannel(attack_rate=0.1, mode=mode),
        )
        value = float(result.log.decision_values.mean())
        rows.append(
            {
                "feedback": "10% adversarial",
                "defenses": label,
                "mean_decision_value": round(value, 4),
                "value_drop": round(baseline - value, 4),
            }
        )
    return rows


def _service_section(environment: SimulatedEnvironment, workdir: Path) -> dict[str, Any]:
    """Real requests through ``RouterService`` on the offline provider.

    ``tenant_small`` is capped low on purpose so the budget guard visibly refuses it, and
    ``tenant_main`` is capped high enough that the rest of the section -- DuckDB logging, feedback
    ingestion, the policy learning from it, a registry snapshot of the live policy -- actually runs.

    The two tenants are driven in separate loops and the small one stops at its first refusal. Doing
    it the other way round -- interleaving them and pushing on -- produced one ``no_arm_available``
    warning per attempt, so the first thing a reader saw was a wall of 27 identical log lines. The
    refusal is the point; repeating it 27 times is not.
    """
    # Only digests of these reach the settings; neither is a credential for anything.
    demo_route_key = "demo-route-key"
    demo_feedback_key = "demo-feedback-key"
    settings = Settings(
        _env_file=None,
        data_dir=workdir / "live",
        environment="test",
        route_api_key_sha256=[hash_api_key(demo_route_key)],
        feedback_api_key_sha256=[hash_api_key(demo_feedback_key)],
        tenant_budget_usd=1.0,
        tenant_budget_caps={"tenant_small": 0.004},
    )
    service = build_router_service(
        settings,
        seed=_SEED,
        workload_digest=environment.workload.digest,
        simulator_seed=_SEED,
    )
    verifier = ScopedApiKeyVerifier.from_settings(settings)
    route_principal = verifier.verify(demo_route_key)
    feedback_principal = ScopedPrincipal(key_id="rubric_a", scopes=frozenset({Scope.FEEDBACK}))
    second_source = ScopedPrincipal(key_id="rubric_b", scopes=frozenset({Scope.FEEDBACK}))
    if route_principal is None:  # pragma: no cover - the key was just minted from this digest
        raise RuntimeError("demo wiring: the generated route key did not verify")
    pool = environment.workload.pool

    def route_one(index: int, tenant: str) -> str | None:
        request = pool[index]
        try:
            response = service.route(
                RouteRequest(
                    tenant_id=tenant,
                    prompt=request.context.prompt,
                    max_output_tokens=request.context.max_output_tokens,
                    tier=request.context.tier,
                ),
                principal=route_principal,
            )
        except Exception:  # noqa: BLE001 - budget refusals are the point of this section
            return None
        return response.decision_id

    decisions: list[str] = []
    for index in range(90):
        decision_id = route_one(index, "tenant_main")
        if decision_id is not None:
            decisions.append(decision_id)
    small_served = 0
    small_refused_after: int | None = None
    for index in range(90, 120):
        decision_id = route_one(index, "tenant_small")
        if decision_id is None:
            small_refused_after = small_served
            break
        small_served += 1
        decisions.append(decision_id)
    served = len(decisions)
    denied = 0 if small_refused_after is None else 1
    applied = 0
    for position, decision_id in enumerate(decisions[:60]):
        reward = 0.4 + 0.4 * ((position * 7) % 5) / 4.0
        for source_index, principal in enumerate((feedback_principal, second_source)):
            result = service.submit_feedback(
                FeedbackSubmission(
                    report_id=f"rep-{position}-{source_index}",
                    decision_id=decision_id,
                    reward=reward if source_index == 0 else reward + 0.02,
                ),
                principal=principal,
            )
            applied += 1 if result.applied_to_policy else 0
    registered_digest = service.register_live_policy(
        metrics={"policy_updates": float(service.policy.updates)}
    )
    summary = {
        "requests_served": served,
        "requests_denied_by_budget": denied,
        "tenant_small_served_before_refusal": small_served,
        "tenant_small_refused_after_requests": small_refused_after,
        "live_policy_snapshot_digest": registered_digest[:16],
        "decisions_logged": service.store.count_decisions(),
        "rewards_aggregated": service.store.count_rewards(),
        "policy_updates": service.policy.updates,
        "feedback_applied_to_policy": applied,
        "spend_by_tenant_usd": {
            tenant: round(micros / 1_000_000, 6)
            for tenant, micros in service.store.spend_by_tenant().items()
        },
        "arm_counts": service.store.arm_counts(),
        "registered_policy_versions": len(service.registry.versions(POLICY_MODEL_NAME)),
        "spend_cap_usd_by_tenant": {"tenant_main": 1.0, "tenant_small": 0.004},
    }
    service.store.close()
    return summary


def run_demo(workdir: Path) -> dict[str, Any]:
    """Run the whole demo, printing tables, and return a JSON-safe summary."""
    started = time.perf_counter()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    workload = Workload(seed=_SEED, pool_size=3_000)
    environment = SimulatedEnvironment(workload=workload, catalog=catalog)

    print("bandit-llm-router: one simulated day, fully offline")
    print(f"workload digest {workload.digest[:16]}  environment digest {environment.digest[:16]}")
    print(f"arms: {', '.join(f'{s.name}={s.model}' for s in catalog.specs)}")

    print("\n## 1. Learning: policies over the same request stream")
    learning_rows, learning_summary = _learning_section(environment)
    print(markdown_table(learning_rows))

    print("\n## 2. Adaptation after a mid-day shift (Haiku degrades on every family)")
    shift_rows, shift_summary = _shift_section(catalog, workload)
    print(markdown_table(shift_rows))

    print("\n## 3. Off-policy evaluation against the simulator's exact values")
    accuracy_rows, dr_rows = _ope_section(environment)
    print(markdown_table(accuracy_rows))
    print()
    print(markdown_table([{k: v for k, v in row.items() if k != "seed"} for row in dr_rows]))

    print("\n## 4. Screening, approval and activation")
    screening_rows, activation_summary = _screening_section(environment, workdir)
    print(markdown_table(screening_rows))
    print(json.dumps(activation_summary, indent=2, sort_keys=True))

    print("\n## 5. Adversarial feedback: 10% of decisions attacked")
    poison_rows = _poisoning_section(environment)
    print(markdown_table(poison_rows))

    print("\n## 6. The live path with the offline provider (DuckDB, budgets, feedback)")
    service_summary = _service_section(environment, workdir)
    print(json.dumps(service_summary, indent=2, sort_keys=True))

    exact_linucb = _trained(environment, "linucb", _SEED + 9)
    stream = workload.sample_indices(2_000, np.random.default_rng(5))
    tracking_metrics = {
        "linucb_true_value": true_policy_value(
            environment, exact_linucb, stream, rng=np.random.default_rng(1)
        ),
        "linucb_expected_cost_micros": expected_policy_cost(
            environment, exact_linucb, stream, rng=np.random.default_rng(2)
        ),
        "linucb_saving_vs_sonnet": float(learning_summary["linucb_saving_vs_sonnet"]),
    }
    run_id = log_policy_run(
        workdir / "mlflow",
        run_name="demo",
        params={
            "workload_digest": workload.digest,
            "environment_digest": environment.digest,
            "policy_version": exact_linucb.version,
            "propensity_floor": _FLOOR,
            "reward_cost_weight": environment.reward_config.cost_weight,
            "simulator_seed": _SEED,
        },
        metrics=tracking_metrics,
        tags={"section": "demo"},
    )
    print(f"\n## 7. MLflow run {run_id} logged to {workdir / 'mlflow'}")

    elapsed = time.perf_counter() - started
    print(f"\ndemo finished in {elapsed:.1f}s")
    return {
        "workload_digest": workload.digest,
        "environment_digest": environment.digest,
        "learning": learning_summary,
        "adaptation": shift_summary,
        "ope": {row["estimator"]: row for row in accuracy_rows},
        "activation": activation_summary,
        "poisoning": poison_rows,
        "service": service_summary,
        "mlflow_run_id": run_id,
        "seconds": round(elapsed, 1),
    }
