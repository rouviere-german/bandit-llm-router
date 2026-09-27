"""Router benchmark: adaptation after a shift, robustness to hostile feedback, and live-path cost.

Three measurements the other benches do not cover:

* **Adaptation.** How many requests a policy needs to get back to 95% of the reference reward after
    a
  scheduled quality shift, discounted posteriors against stationary ones. The reference is the
  clairvoyant policy *with the same propensity floor*, not the unconstrained oracle: the floor's
  fixed
  exploration cost is a design choice, and charging a policy for it would make the target
  unreachable
  for reasons that have nothing to do with learning.
* **Robustness.** Policy value under adversarial feedback, in four configurations: no defenses;
  clipping only; the full aggregation (clip, per-source median, weight-capped weighted median, trust
  weighting); and the channel the **online posterior** actually gets, which is the first accepted
  report for a decision, clipped, at the reduced single-source weight. The same poisoned stream
  drives all four, so the difference is the defenses and nothing else. The last row is here because
  the full aggregation protects the *stored* reward -- what OPE and offline retraining read -- while
  ``RouterService.submit_feedback`` teaches the live policy once per decision, from one report, so
  the median has nothing to combine. Publishing only the full-pipeline number would overstate what
  the deployed learner is protected by.
* **Live-path throughput.** Wall-clock per request through ``RouterService`` on the offline
    provider:
  featurize, decide, restrict, reserve, call, commit, log to DuckDB. Not a model-latency number --
  it is the routing overhead, which is the part this repo is responsible for.

Usage:
  uv run python bench/bench_router.py            # run and write bench/results/router.json
  uv run python bench/bench_router.py --check    # also enforce bench/thresholds.json
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# bench/ is a script directory on sys.path, not an installed package.
from _common import ROOT, add_common_arguments, machine, threshold_problems, write_results
from bandit_llm_router.arms import load_catalog
from bandit_llm_router.auth import Scope, ScopedPrincipal, hash_api_key
from bandit_llm_router.budgets import TenantBudgetGuard
from bandit_llm_router.core.config import Settings
from bandit_llm_router.core.logging import configure_logging
from bandit_llm_router.experiments import (
    AggregationMode,
    PoisonedFeedbackChannel,
    ShiftRecovery,
    run_shift_experiment,
    run_simulation,
)
from bandit_llm_router.features import Featurizer
from bandit_llm_router.policies.linear_bandit import LinearBanditPolicy, LinUCB
from bandit_llm_router.reporting import markdown_table
from bandit_llm_router.service import RouteRequest, build_router_service
from bandit_llm_router.simulator import QualityShift, SimulatedEnvironment, Workload
from bandit_llm_router.simulator.workload import FAMILIES

FloatArray = npt.NDArray[np.float64]

SEED = 20_260_927
POOL_SIZE = 4_096
FLOOR = 0.02
DEFAULT_SHIFT_REQUESTS = 120_000
DEFAULT_POISON_REQUESTS = 40_000
DEFAULT_SERVICE_REQUESTS = 2_000
ATTACK_RATE = 0.10
RECOVERY_TARGET = 0.95


def policy(environment: SimulatedEnvironment, *, discount: float) -> LinearBanditPolicy:
    return LinUCB(
        arms=environment.arms,
        dim=environment.feature_dim,
        feature_version=environment.featurizer.version,
        propensity_floor=FLOOR,
        alpha=0.5,
        discount=discount,
    )


# Sentinel for "the shift knocked this policy below target and it never got back". It has to be a
# number because every threshold in bench/thresholds.json is a numeric upper bound, and it has to be
# far above any plausible recovery count so it can only ever read as a failure.
_NEVER_RECOVERED = 10.0**9
# How far below its pre-shift reward ratio the stationary policy has to be driven for the adaptation
# comparison to mean anything. Measured at ~0.42 on this workload; 0.15 leaves room for noise while
# still failing loudly if a simulator change makes the shift toothless.
_MIN_STATIONARY_TROUGH_DEPTH = 0.15


def _recovery_cost(recovery: ShiftRecovery) -> float:
    """Requests to recover, as one comparable number, lowest is best.

    Three outcomes collapse into this scale, and conflating the first with the last is the mistake
    worth guarding against:

    * **never dipped below target** -- the best possible outcome, scored ``0``. ``ShiftRecovery``
      reports this as ``recovery_requests == 0`` with ``dipped_below_target`` false, so a plain
      ``recovery_requests or SENTINEL`` silently turns the best result into the worst one (``0`` is
      falsy). That is exactly what this helper exists to prevent.
    * **dipped and recovered after k requests** -- scored ``k``.
    * **dipped and never recovered** -- scored :data:`_NEVER_RECOVERED`.
    """
    if not recovery.dipped_below_target:
        return 0.0
    if recovery.recovery_requests is None:
        return _NEVER_RECOVERED
    return float(recovery.recovery_requests)


def _never_recovered(recovery: ShiftRecovery) -> float:
    """1.0 only when the shift pushed the policy below target and it stayed there."""
    return 1.0 if recovery.dipped_below_target and recovery.recovery_requests is None else 0.0


def run_adaptation(
    *, requests: int, featurizer: Featurizer, workload: Workload, catalog: Any
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """A mid-run model update that degrades the arm the router had learned to prefer.

    Degrading one arm on one prompt family is the realistic version and it is invisible at this
    scale: it moves a few percent of traffic and both posteriors shrug it off, reporting zero
    requests
    to recover. Haiku carries roughly three quarters of the decisions, so degrading *it* across
    every
    family is what turns "requests to recover" into a number worth reporting.
    """
    shift_step = requests // 3
    shifted = SimulatedEnvironment(
        workload=workload,
        catalog=catalog,
        featurizer=featurizer,
        shifts=tuple(
            QualityShift(at_step=shift_step, family=family, arm="haiku", intercept_delta=-3.5)
            for family in FAMILIES
        ),
    )
    recoveries = run_shift_experiment(
        shifted,
        policies={
            "stationary": (1.0, policy(shifted, discount=1.0)),
            "discounted_0.999": (0.999, policy(shifted, discount=0.999)),
            "discounted_0.9995": (0.9995, policy(shifted, discount=0.9995)),
        },
        n=requests,
        seed=SEED,
        shift_step=shift_step,
        window=requests // 12,
        target=RECOVERY_TARGET,
    )
    rows = [item.summary_row() for item in recoveries]
    by_name = {item.policy_name: item for item in recoveries}
    stationary = by_name["stationary"]
    best_discounted = min(
        (by_name["discounted_0.999"], by_name["discounted_0.9995"]), key=_recovery_cost
    )
    measurements = {
        "adaptation": {
            # Thresholds are upper bounds, so "recovered at all" is expressed as a flag that must
            # stay at zero rather than as a lower bound on something.
            "stationary_never_recovered": _never_recovered(stationary),
            "discounted_never_recovered": _never_recovered(best_discounted),
            "discounted_recovery_requests": _recovery_cost(best_discounted),
            # Forgetting should pay for itself: the discounted policy must not be the slower one.
            "discounted_slower_than_stationary": 0.0
            if _recovery_cost(best_discounted) <= _recovery_cost(stationary)
            else 1.0,
            "trough_depth_stationary": round(
                stationary.pre_shift_ratio - stationary.post_shift_trough, 4
            ),
            # The whole comparison is vacuous if the shift stops hurting the stationary policy: it
            # would "recover in 0 requests" and the discounted variant could never look better.
            # Recorded as a shortfall so it reads as an upper bound like every other threshold.
            "trough_shortfall_stationary": round(
                max(
                    0.0,
                    _MIN_STATIONARY_TROUGH_DEPTH
                    - (stationary.pre_shift_ratio - stationary.post_shift_trough),
                ),
                4,
            ),
        }
    }
    return measurements, rows


def run_poisoning(
    environment: SimulatedEnvironment, *, requests: int
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """The same hostile stream with the defenses off, half on and fully on."""
    stream = environment.workload.sample_indices(requests, np.random.default_rng(SEED + 31))
    clean = run_simulation(
        environment, policy(environment, discount=1.0), n=requests, seed=SEED, stream=stream
    )
    baseline = float(clean.log.decision_values.mean())
    rows: list[dict[str, Any]] = [
        {
            "feedback": "honest only",
            "defenses": "n/a",
            "mean_decision_value": round(baseline, 4),
            "value_drop": 0.0,
        }
    ]
    drops: dict[str, float] = {}
    ablation: tuple[tuple[AggregationMode, str], ...] = (
        ("none", "none"),
        ("clip_only", "clipping only"),
        ("robust", "full aggregation (stored reward)"),
        ("first_report_only", "online posterior (first report, clipped)"),
    )
    for mode, label in ablation:
        result = run_simulation(
            environment,
            policy(environment, discount=1.0),
            n=requests,
            seed=SEED,
            stream=stream,
            channel=PoisonedFeedbackChannel(attack_rate=ATTACK_RATE, mode=mode),
        )
        value = float(result.log.decision_values.mean())
        drops[mode] = baseline - value
        rows.append(
            {
                "feedback": f"{ATTACK_RATE:.0%} adversarial",
                "defenses": label,
                "mean_decision_value": round(value, 4),
                "value_drop": round(baseline - value, 4),
            }
        )
    measurements = {
        "poisoning": {
            "value_drop_undefended": round(drops["none"], 4),
            "value_drop_clip_only": round(drops["clip_only"], 4),
            "value_drop_defended": round(drops["robust"], 4),
            # What the deployed online learner gets, which is weaker than the row above it.
            "value_drop_online_path": round(drops["first_report_only"], 4),
            # The defenses must actually be worth something.
            "defenses_no_better_than_none": 0.0 if drops["robust"] < drops["none"] else 1.0,
            # And the honest ordering has to hold: one clipped report cannot beat three
            # corroborated ones. If this flips, the online row stopped measuring the weaker channel.
            "online_path_not_weaker_than_full": 0.0
            if drops["first_report_only"] > drops["robust"]
            else 1.0,
        }
    }
    return measurements, rows


def _time_route(
    environment: SimulatedEnvironment, *, requests: int, shared_budget_file: bool
) -> tuple[FloatArray, int, dict[str, int]]:
    route_key = "bench-route-key"
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,
            environment="test",
            data_dir=Path(tmp) / "var",
            pricing_path=ROOT / "config" / "pricing.toml",
            route_api_key_sha256=[hash_api_key(route_key)],
            propensity_floor=FLOOR,
            tenant_budget_usd=10_000.0,
        )
        service = build_router_service(
            settings,
            seed=SEED,
            workload_digest=environment.workload.digest,
            simulator_seed=SEED,
        )
        if not shared_budget_file:
            service.guard = TenantBudgetGuard(service.guard.budgets, path=None)
        principal = ScopedPrincipal(key_id="bench", scopes=frozenset({Scope.ROUTE}))
        pool = environment.workload.pool
        latencies = np.empty(requests, dtype=np.float64)
        for index in range(requests):
            request = pool[index % len(pool)]
            body = RouteRequest(
                tenant_id="tenant_bench",
                prompt=request.context.prompt,
                max_output_tokens=request.context.max_output_tokens,
                tier=request.context.tier,
            )
            start = time.perf_counter_ns()
            service.route(body, principal=principal)
            latencies[index] = (time.perf_counter_ns() - start) / 1_000.0
        logged = service.store.count_decisions()
        arm_counts = service.store.arm_counts()
        service.store.close()
    return latencies, logged, arm_counts


def run_service(
    environment: SimulatedEnvironment, *, requests: int
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Routing overhead per request on the live path, offline provider, DuckDB logging on.

    Measured twice, because one number here would hide the interesting half. The default guard keeps
    every tenant's spend in a lock-protected JSON file so several workers share one cap instead of
    each spending the full amount -- and it pays for that with two atomic writes and a reload per
    reserve and per commit. Running the same path with an in-memory guard isolates the decision work
    (featurize, decide, restrict, call, log) from the cost of the shared cap, so the README can say
    which is which instead of quoting one opaque figure. ADR 0006 records the trade-off.
    """
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}
    for label, shared in (("shared budget file", True), ("in-memory budget guard", False)):
        latencies, logged, arm_counts = _time_route(
            environment, requests=requests, shared_budget_file=shared
        )
        rows.append(
            {
                "budget guard": label,
                "requests": requests,
                "decisions_logged": logged,
                "p50_us": round(float(np.percentile(latencies, 50)), 1),
                "p95_us": round(float(np.percentile(latencies, 95)), 1),
                "p99_us": round(float(np.percentile(latencies, 99)), 1),
                "arm_counts": tuple(f"{arm}={count}" for arm, count in sorted(arm_counts.items())),
            }
        )
        key = "route_decision_shared_budget" if shared else "route_decision_in_memory_budget"
        measurements[key] = {
            "p95_us": round(float(np.percentile(latencies, 95)), 1),
            "lost_decisions": float(requests - logged),
        }
    return measurements, rows


def run(*, shift_requests: int, poison_requests: int, service_requests: int) -> dict[str, Any]:
    started = time.perf_counter()
    catalog = load_catalog(ROOT / "config" / "pricing.toml")
    featurizer = Featurizer()
    workload = Workload(seed=SEED, pool_size=POOL_SIZE)
    environment = SimulatedEnvironment(workload=workload, catalog=catalog, featurizer=featurizer)

    adaptation_measurements, adaptation_rows = run_adaptation(
        requests=shift_requests, featurizer=featurizer, workload=workload, catalog=catalog
    )
    poisoning_measurements, poisoning_rows = run_poisoning(environment, requests=poison_requests)
    service_measurements, service_rows = run_service(environment, requests=service_requests)

    return {
        "machine": machine(),
        "config": {
            "shift_requests": shift_requests,
            "poison_requests": poison_requests,
            "service_requests": service_requests,
            "attack_rate": ATTACK_RATE,
            "recovery_target": RECOVERY_TARGET,
            "propensity_floor": FLOOR,
            "workload_digest": workload.digest,
            "environment_digest": environment.digest,
        },
        "results": {
            **adaptation_measurements,
            **poisoning_measurements,
            **service_measurements,
        },
        "tables": {
            "adaptation": adaptation_rows,
            "poisoning": poisoning_rows,
            "service": service_rows,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="router adaptation, robustness and overhead")
    add_common_arguments(parser)
    parser.add_argument("--shift-requests", type=int, default=DEFAULT_SHIFT_REQUESTS)
    parser.add_argument("--poison-requests", type=int, default=DEFAULT_POISON_REQUESTS)
    parser.add_argument("--service-requests", type=int, default=DEFAULT_SERVICE_REQUESTS)
    args = parser.parse_args(argv)

    # The routing path logs one line per decision; stdout is the report, so those go to stderr.
    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    payload = run(
        shift_requests=args.shift_requests,
        poison_requests=args.poison_requests,
        service_requests=args.service_requests,
    )
    write_results("router.json", payload, reference=args.update_reference)

    print(f"# adaptation after a mid-run shift ({args.shift_requests:,} requests)")
    print(markdown_table(payload["tables"]["adaptation"]))
    print(f"\n# {ATTACK_RATE:.0%} adversarial feedback ({args.poison_requests:,} requests)")
    print(markdown_table(payload["tables"]["poisoning"]))
    print(f"\n# routing overhead on the live path ({args.service_requests:,} requests)")
    print(markdown_table(payload["tables"]["service"]))
    print(f"\nmeasured in {payload['seconds']}s on {payload['machine']['cpu_count']} CPUs")

    if args.check:
        problems = threshold_problems("bench_router", payload["results"])
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all router benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
