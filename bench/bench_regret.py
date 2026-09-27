"""Learning benchmark: cumulative regret and cost against the static baselines.

What is measured, and why each number is the one the README quotes:

* **In-class regret slope.** Regret against the clairvoyant policy *with the same propensity floor*,
  fitted as the slope of ``log R(t)`` on ``log t`` over the run's tail. This is the number the
  thresholds track, because it isolates learning: the floor spends a fixed fraction of traffic on
  exploration forever, so regret against the *unconstrained* oracle cannot be sublinear no matter
  how
  well a policy learns. Oracle regret is reported next to it so the gap is visible.
* **Realizable regret slope.** The same slope on a bandit whose expected reward really is linear in
  the features. That is the setting LinUCB and Thompson sampling are analysed in, so a sublinear
  rate
  there is a claim about this implementation rather than about the simulator. The production
  simulator
  deliberately violates realizability (sigmoid quality over a latent difficulty), and its slope is
  reported separately.
* **Cost against always-Sonnet**, at the quality each policy actually achieved. "30% cheaper" is
  meaningless without saying cheaper than what and at what quality, so both are in the table.

Usage:
  uv run python bench/bench_regret.py               # run and write bench/results/regret.json
  uv run python bench/bench_regret.py --check       # also enforce bench/thresholds.json
  uv run python bench/bench_regret.py --requests N  # override the request count
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

# bench/ is a script directory on sys.path, not an installed package.
from _common import ROOT, machine, threshold_problems, write_results
from bandit_llm_router.arms import ArmCatalog, load_catalog
from bandit_llm_router.experiments import (
    SimulationResult,
    run_linear_bandit,
    run_simulation,
)
from bandit_llm_router.features import Featurizer
from bandit_llm_router.policies.baselines import KeywordRule, StaticArm
from bandit_llm_router.policies.linear_bandit import (
    EpsilonGreedy,
    LinearThompson,
    LinUCB,
)
from bandit_llm_router.reporting import markdown_table
from bandit_llm_router.simulator import SimulatedEnvironment, Workload

DEFAULT_REQUESTS = 200_000
POOL_SIZE = 8_192
SEED = 20_260_927
FLOOR = 0.02
LEARNERS = ("linucb", "thompson", "epsilon_greedy")
BASELINES = ("keyword_rule", "haiku", "sonnet", "opus")


def build_policy(kind: str, environment: SimulatedEnvironment) -> Any:
    arms, dim = environment.arms, environment.feature_dim
    version = environment.featurizer.version
    common: dict[str, Any] = {
        "arms": arms,
        "dim": dim,
        "feature_version": version,
        "propensity_floor": FLOOR,
    }
    if kind == "linucb":
        return LinUCB(alpha=0.5, **common)
    if kind == "thompson":
        return LinearThompson(posterior_sd=0.3, mc_samples=24, **common)
    if kind == "epsilon_greedy":
        return EpsilonGreedy(epsilon=0.05, **common)
    if kind == "keyword_rule":
        return KeywordRule(
            arms=arms,
            dim=dim,
            feature_version=version,
            code_arm="opus",
            long_arm="sonnet",
            default_arm="haiku",
            propensity_floor=FLOOR,
        )
    return StaticArm(
        arms=arms, dim=dim, feature_version=version, target=kind, propensity_floor=FLOOR
    )


def run_simulator_benchmark(
    environment: SimulatedEnvironment, *, requests: int
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Every policy over the same request stream, so the comparison is about the decision rule."""
    stream = environment.workload.sample_indices(requests, np.random.default_rng(SEED + 1))
    results: dict[str, SimulationResult] = {}
    for kind in (*LEARNERS, *BASELINES):
        results[kind] = run_simulation(
            environment,
            build_policy(kind, environment),
            n=requests,
            seed=SEED,
            learn=kind in LEARNERS,
            stream=stream,
        )
    sonnet = results["sonnet"]
    opus = results["opus"]
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}
    for kind, result in results.items():
        saving_vs_sonnet = 1.0 - result.mean_cost_micros / sonnet.mean_cost_micros
        saving_vs_opus = 1.0 - result.mean_cost_micros / opus.mean_cost_micros
        rows.append(
            {
                "policy": kind,
                "mean_reward": round(result.mean_reward, 4),
                "mean_quality": round(result.mean_quality, 4),
                "usd_per_1k_requests": round(result.mean_cost_micros / 1_000.0, 3),
                "saving_vs_sonnet": round(saving_vs_sonnet, 3),
                "saving_vs_opus": round(saving_vs_opus, 3),
                "quality_delta_vs_sonnet": round(result.mean_quality - sonnet.mean_quality, 4),
                "oracle_regret": round(result.final_regret, 1),
                "oracle_slope": round(result.regret_slope, 3),
                "in_class_regret": round(result.final_class_regret, 1),
                "in_class_slope": round(result.class_regret_slope, 3),
                "min_propensity": round(result.min_propensity, 4),
                "arm_shares": tuple(round(share, 3) for share in result.arm_shares),
            }
        )
        measurements[f"simulator_{kind}"] = {
            "in_class_slope": round(result.class_regret_slope, 4),
            "oracle_slope": round(result.regret_slope, 4),
            "mean_reward": round(result.mean_reward, 4),
            "usd_per_1k_requests": round(result.mean_cost_micros / 1_000.0, 4),
            # Thresholds are one-sided upper bounds, so a *shortfall* is expressed as a number that
            # must stay small: the amount by which the saving misses 30%.
            "saving_shortfall_vs_sonnet": round(max(0.0, 0.30 - saving_vs_sonnet), 4),
            "quality_shortfall_vs_sonnet": round(
                max(0.0, sonnet.mean_quality - result.mean_quality), 4
            ),
        }
    return measurements, rows


def run_realizable_benchmark(
    *, requests: int
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """A well-specified linear bandit: the only place a regret *rate* is a claim about the code."""
    rng = np.random.default_rng(SEED + 7)
    dim = 8
    features = np.column_stack(
        [np.ones(2_048), rng.uniform(-1.0, 1.0, size=(2_048, dim - 1)) * 0.5]
    )
    thetas = np.vstack(
        [
            np.concatenate([[0.45], rng.uniform(-0.25, 0.35, size=dim - 1)]),
            np.concatenate([[0.50], rng.uniform(-0.25, 0.35, size=dim - 1)]),
            np.concatenate([[0.48], rng.uniform(-0.25, 0.35, size=dim - 1)]),
        ]
    )
    arms = ("a", "b", "c")
    policies: dict[str, Any] = {
        "linucb": LinUCB(arms=arms, dim=dim, alpha=0.4, propensity_floor=FLOOR),
        "thompson": LinearThompson(
            arms=arms, dim=dim, posterior_sd=0.25, mc_samples=24, propensity_floor=FLOOR
        ),
        "epsilon_greedy": EpsilonGreedy(arms=arms, dim=dim, epsilon=0.05, propensity_floor=FLOOR),
        "static_a": StaticArm(arms=arms, dim=dim, target="a", propensity_floor=FLOOR),
    }
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        result = run_linear_bandit(
            policy, features=features, thetas=thetas, n=requests, seed=SEED + 11
        )
        rows.append(
            {
                "policy": name,
                "n": result.n,
                "regret": round(result.final_regret, 1),
                "regret_slope": round(result.regret_slope, 3),
            }
        )
        measurements[f"realizable_{name}"] = {
            "regret_slope": round(result.regret_slope, 4),
            "regret": round(result.final_regret, 2),
        }
    return measurements, rows


def run(*, requests: int, realizable_requests: int) -> dict[str, Any]:
    started = time.perf_counter()
    catalog: ArmCatalog = load_catalog(ROOT / "config" / "pricing.toml")
    featurizer = Featurizer()
    workload = Workload(seed=SEED, pool_size=POOL_SIZE)
    environment = SimulatedEnvironment(workload=workload, catalog=catalog, featurizer=featurizer)
    simulator_measurements, simulator_rows = run_simulator_benchmark(environment, requests=requests)
    realizable_measurements, realizable_rows = run_realizable_benchmark(
        requests=realizable_requests
    )
    return {
        "machine": machine(),
        "config": {
            "requests": requests,
            "realizable_requests": realizable_requests,
            "pool_size": POOL_SIZE,
            "seed": SEED,
            "propensity_floor": FLOOR,
            "workload_digest": workload.digest,
            "environment_digest": environment.digest,
            "featurizer_version": featurizer.version,
        },
        "results": {**simulator_measurements, **realizable_measurements},
        "tables": {"simulator": simulator_rows, "realizable": realizable_rows},
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="learning and cost benchmark")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--realizable-requests", type=int, default=60_000)
    args = parser.parse_args(argv)

    payload = run(requests=args.requests, realizable_requests=args.realizable_requests)
    write_results("regret.json", payload)

    print(f"# learning over {payload['config']['requests']:,} simulated requests")
    print(f"workload digest {payload['config']['workload_digest'][:16]}")
    print()
    print(markdown_table(payload["tables"]["simulator"]))
    print()
    print(f"# a well-specified linear bandit ({args.realizable_requests:,} requests)")
    print(markdown_table(payload["tables"]["realizable"]))
    print(f"\nmeasured in {payload['seconds']}s on {payload['machine']['cpu_count']} CPUs")

    if args.check:
        problems = threshold_problems("bench_regret", payload["results"])
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all learning benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
