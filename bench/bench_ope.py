"""Off-policy evaluation benchmark: bias, RMSE and interval coverage against the exact truth.

The simulator knows every arm's expected reward, so a frozen policy's value is computed exactly
rather than estimated. That exact value is the ground truth each estimator is scored against, which
is what makes "DR is more accurate than IPS here" a measurement instead of an opinion.

Four things this bench is careful about:

* **The accuracy sweeps freeze the logging policy.** An online learner's log is not i.i.d., so a
  bootstrap interval over its rows would not be measuring what it claims to. Freezing costs a short
  warm-up and makes the reported coverage meaningful.
* **The shipped gate does not get to freeze anything, so that case is measured too.**
  ``RouterService.screen`` evaluates candidates on the log its own learning policy wrote. The last
  section logs the *same* candidates and seeds twice -- once frozen, once with the logger still
  updating -- and reports DR coverage under the row bootstrap and under a block bootstrap over
  contiguous rows. The four numbers say how far the headline coverage above transfers to the
  governance path.
* **Coverage is measured, not assumed.** The DR interval is a percentile bootstrap over rows with
    the
  reward model held fixed, so it covers sampling noise in the log but not the reward model's own
  uncertainty. The empirical 95% coverage is reported, and the thresholds bound how far it may drift
  from nominal in *either* direction -- an interval that is too wide is also wrong.
* **Small propensities are exercised on purpose.** A second sweep logs with a low floor, which is
  where importance weighting hurts most; that is the regime the screening gate's ESS check exists
  for.

Usage:
  uv run python bench/bench_ope.py            # run and write bench/results/ope.json
  uv run python bench/bench_ope.py --check    # also enforce bench/thresholds.json
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# bench/ is a script directory on sys.path, not an installed package.
from _common import ROOT, add_common_arguments, machine, threshold_problems, write_results
from bandit_llm_router.arms import load_catalog
from bandit_llm_router.experiments import (
    AdaptiveCoverage,
    CandidateFactory,
    OPEAccuracy,
    run_adaptive_coverage,
    run_ope_accuracy,
    run_screening_trials,
    run_simulation,
)
from bandit_llm_router.features import Featurizer
from bandit_llm_router.ope.screening import ScreeningConfig
from bandit_llm_router.policies.baselines import KeywordRule, StaticArm
from bandit_llm_router.policies.linear_bandit import (
    EpsilonGreedy,
    LinearBanditPolicy,
    LinearThompson,
    LinUCB,
)
from bandit_llm_router.reporting import markdown_table
from bandit_llm_router.simulator import SimulatedEnvironment, Workload

SEED = 20_260_927
POOL_SIZE = 4_096
DEFAULT_LOGS = 40_000
WARMUP = 4_000
BOOTSTRAP = 300
# Coverage is a proportion, so it needs trials to be worth quoting: at 4 candidates per seed, 15
# seeds give 60 trials and a standard error near 0.03, which can tell 0.95 from 0.80. Five seeds
# would report "1.00" for any true coverage above about 0.86 -- a number that says nothing.
SEEDS = tuple(range(1, 16))
SCREENING_SEEDS = (1, 2, 3)
NOMINAL_COVERAGE = 0.95
# Block length for the adaptive-log bootstrap. It has to be long enough to contain the serial
# dependence a learner induces and short enough to leave many blocks: at 40,000 rows, 250 leaves
# 160. This is the value `screening_bootstrap_block_size` ships with, so the bench measures the
# default rather than a number chosen to look good.
BLOCK_SIZE = 250
# One logging run per seed is shared by both bootstraps, so 15 seeds x 4 candidates = 60 trials per
# cell, the same count as the frozen sweeps above -- the comparison is not between sample sizes.
ADAPTIVE_SEEDS = tuple(range(1, 16))
# The adaptive section uses its own, shorter log on purpose and does *not* follow --logs. A LinUCB
# on this workload has largely converged after a few thousand requests, so in a 40,000-row window
# most rows are written by a policy that has stopped moving -- which dilutes the very effect being
# measured. The arm-share drift column reports what is left of it either way, and the threshold
# `adaptive_setup.online_log_barely_moved` fails the bench if the online log stops being adaptive.
ADAPTIVE_LOGS = 8_000


def learner(
    environment: SimulatedEnvironment, kind: str, *, floor: float, **overrides: Any
) -> LinearBanditPolicy:
    common: dict[str, Any] = {
        "arms": environment.arms,
        "dim": environment.feature_dim,
        "feature_version": environment.featurizer.version,
        "propensity_floor": floor,
    }
    common.update(overrides)
    if kind == "linucb":
        return LinUCB(**{"alpha": 0.5, **common})
    if kind == "thompson":
        return LinearThompson(**{"posterior_sd": 0.3, "mc_samples": 24, **common})
    return EpsilonGreedy(**{"epsilon": 0.1, **common})


def static_candidate(target: str, floor: float) -> CandidateFactory:
    return lambda env, _seed: StaticArm(
        arms=env.arms,
        dim=env.feature_dim,
        target=target,
        propensity_floor=floor,
        feature_version=env.featurizer.version,
    )


def rule_candidate(floor: float) -> CandidateFactory:
    return lambda env, _seed: KeywordRule(
        arms=env.arms,
        dim=env.feature_dim,
        code_arm="opus",
        long_arm="sonnet",
        default_arm="haiku",
        propensity_floor=floor,
        feature_version=env.featurizer.version,
    )


def trained_candidate(kind: str, floor: float) -> CandidateFactory:
    def build(env: SimulatedEnvironment, seed: int) -> LinearBanditPolicy:
        policy = learner(env, kind, floor=floor)
        run_simulation(env, policy, n=WARMUP, seed=seed + 101, learn=True)
        return policy

    return build


def candidates(floor: float) -> dict[str, CandidateFactory]:
    """A spread of candidates: two static arms, a hand-written rule and a trained learner.

    The static arms are the hard cases -- they concentrate on one action, so importance weights
    spike
    and ESS collapses, which is exactly where the three estimators diverge.
    """
    return {
        "always_haiku": static_candidate("haiku", floor),
        "always_opus": static_candidate("opus", floor),
        "keyword_rule": rule_candidate(floor),
        "trained_thompson": trained_candidate("thompson", floor),
    }


def worse_candidates(floor: float) -> dict[str, CandidateFactory]:
    """Candidates a converged, cost-aware incumbent genuinely beats."""
    return {
        "always_opus": static_candidate("opus", floor),
        "always_sonnet": static_candidate("sonnet", floor),
        "keyword_rule": rule_candidate(floor),
    }


def better_candidates(floor: float) -> dict[str, CandidateFactory]:
    """Candidates that genuinely beat a stale, cost-blind incumbent."""
    return {
        "always_haiku": static_candidate("haiku", floor),
        "keyword_rule": rule_candidate(floor),
        "trained_thompson": trained_candidate("thompson", floor),
    }


def trained_incumbent(
    environment: SimulatedEnvironment, *, floor: float, seed: int
) -> LinearBanditPolicy:
    """A converged LinUCB: the incumbent a new candidate usually has to beat, and usually cannot."""
    policy = learner(environment, "linucb", floor=floor)
    run_simulation(environment, policy, n=WARMUP * 3, seed=seed + 211, learn=True)
    return policy


def stale_incumbent(environment: SimulatedEnvironment, *, floor: float) -> LinearBanditPolicy:
    """An incumbent from before cost was part of the objective: it always reaches for the big model.

    Note what does *not* work here: an incumbent that always picks the cheapest arm. On this
    workload
    the cheap arm is genuinely the best choice for about three quarters of the traffic, so "always
    cheapest" is a strong policy and nothing can beat it -- which is how the admission side of this
    measurement came out vacuously zero at first. Quality-at-any-price is the honest example of a
    policy a cost-aware router should be able to improve on.

    It is a real LinUCB with real posteriors, taught on the pool's own contexts, so the comparison
    is
    between two policies of the same class rather than against a hand-written stub.
    """
    policy = learner(environment, "linucb", floor=floor, alpha=0.0)
    dearest = int(np.argmax(environment.cost_matrix.mean(axis=0)))
    for row in environment.feature_matrix:
        for arm in range(environment.n_arms):
            policy.update(row, arm, 1.0 if arm == dearest else 0.0)
    return policy


def accuracy_rows(
    label: str, accuracy: dict[str, OPEAccuracy]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}
    for method in ("ips", "snips", "dr"):
        score = accuracy[method]
        rows.append({"logging_floor": label, **score.summary_row()})
        measurements[f"{label}_{method}"] = {
            "abs_bias": round(abs(score.bias), 4),
            "rmse": round(score.rmse, 4),
            # Coverage must stay near nominal from *both* sides, so the metric is the distance from
            # it: an interval that is far too wide is as wrong as one that is too narrow.
            "coverage_gap": round(abs(score.coverage - NOMINAL_COVERAGE), 4),
            "coverage_shortfall": round(max(0.0, 0.90 - score.coverage), 4),
        }
    return rows, measurements


def adaptive_rows(
    coverage: Sequence[AdaptiveCoverage],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}
    for entry in coverage:
        rows.append(entry.summary_row())
        regime = entry.logging_regime.replace(" ", "_")
        block = (
            "rows" if entry.bootstrap_block_size is None else f"block{entry.bootstrap_block_size}"
        )
        measurements[f"adaptive_{regime}_{block}"] = {
            "abs_bias": round(abs(entry.bias), 4),
            "rmse": round(entry.rmse, 4),
            "coverage_shortfall": round(max(0.0, 0.90 - entry.coverage), 4),
            "coverage_gap": round(abs(entry.coverage - NOMINAL_COVERAGE), 4),
        }
    online = [entry for entry in coverage if entry.logging_regime.startswith("online")]
    frozen = [entry for entry in coverage if entry.logging_regime == "frozen"]
    if online and frozen:
        # Without this the whole section could pass while comparing a frozen log with itself.
        measurements["adaptive_setup"] = {
            # Twice the frozen regime's own noise floor. That floor is not zero: comparing the
            # first tenth of a log with the last tenth moves a little on action sampling alone.
            "online_log_barely_moved": 0.0
            if min(entry.logging_drift for entry in online)
            > 2.0 * max(entry.logging_drift for entry in frozen)
            else 1.0
        }
    return rows, measurements


def run(*, n_logs: int) -> dict[str, Any]:
    started = time.perf_counter()
    catalog = load_catalog(ROOT / "config" / "pricing.toml")
    featurizer = Featurizer()
    workload = Workload(seed=SEED, pool_size=POOL_SIZE)
    environment = SimulatedEnvironment(workload=workload, catalog=catalog, featurizer=featurizer)

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, float]] = {}

    def logger_for(floor: float) -> Callable[[], LinearBanditPolicy]:
        return lambda: learner(environment, "epsilon_greedy", floor=floor)

    # A generous floor and a tight one: the second is the regime where importance weighting hurts.
    for label, floor in (("floor_0.05", 0.05), ("floor_0.01", 0.01)):
        accuracy, details = run_ope_accuracy(
            environment,
            logger_factory=logger_for(floor),
            candidates=candidates(floor),
            n_logs=n_logs,
            seeds=SEEDS,
            warmup=WARMUP,
            bootstrap=BOOTSTRAP,
        )
        section_rows, section_measurements = accuracy_rows(label, accuracy)
        rows.extend(section_rows)
        measurements.update(section_measurements)
        detail_rows.extend(
            {"logging_floor": label, **row}
            for row in details
            if row["estimator"] == "dr" and row["seed"] == SEEDS[0]
        )

    # Screening is the consumer of these estimates, so its decision quality is measured here too --
    # in two sweeps, because one incumbent cannot exercise both sides. Against a converged incumbent
    # every candidate is genuinely worse, so only the rejection rate means anything; against a stale
    # one every candidate is genuinely better, so only the admission rate does. Reporting both from
    # a
    # single sweep would make one of the two numbers an artefact of the setup.
    screening_config = ScreeningConfig(
        bootstrap=BOOTSTRAP, min_ess=1_000.0, max_exploration_usd=900.0
    )
    rejection = run_screening_trials(
        environment,
        logger_factory=logger_for(0.05),
        incumbent_factory=lambda env, seed: trained_incumbent(env, floor=0.05, seed=seed),
        candidates=worse_candidates(0.05),
        n_logs=n_logs,
        seeds=SCREENING_SEEDS,
        warmup=WARMUP,
        config=screening_config,
    )
    admission = run_screening_trials(
        environment,
        logger_factory=logger_for(0.05),
        incumbent_factory=lambda env, _seed: stale_incumbent(env, floor=0.05),
        candidates=better_candidates(0.05),
        n_logs=n_logs,
        seeds=SCREENING_SEEDS,
        warmup=WARMUP,
        config=screening_config,
    )
    # What the headline coverage is worth on the log the shipped gate actually screens on. The
    # logging policy keeps learning, so its rows are serially dependent; both bootstraps are then
    # run on identical data so the only difference is the resampling scheme.
    # The online regime starts cold (no warm-up) so the logging policy really is moving while it
    # writes the rows -- which is both the honest worst case and what screening an early policy
    # looks like. `arm-share drift` in the table below is how far it moved.
    adaptive: list[AdaptiveCoverage] = []
    for learning, warmup, label in ((False, WARMUP, "frozen"), (True, 0, "online learner")):
        adaptive.extend(
            run_adaptive_coverage(
                environment,
                logger_factory=logger_for(0.05),
                candidates=candidates(0.05),
                n_logs=ADAPTIVE_LOGS,
                seeds=ADAPTIVE_SEEDS,
                warmup=warmup,
                bootstrap=BOOTSTRAP,
                block_sizes=(None, BLOCK_SIZE),
                learn_while_logging=learning,
                regime_label=label,
            )
        )
    adaptive_table, adaptive_measurements = adaptive_rows(adaptive)
    measurements.update(adaptive_measurements)

    measurements["screening"] = {
        "rejection_shortfall": round(max(0.0, 0.90 - rejection.rejection_rate), 4),
        "admission_shortfall": round(max(0.0, 0.50 - admission.admission_rate), 4),
        # A sweep with nothing of the right kind in it would pass every threshold vacuously.
        "empty_rejection_sweep": 0.0 if rejection.truly_worse >= 3 else 1.0,
        "empty_admission_sweep": 0.0 if admission.truly_better >= 3 else 1.0,
    }

    return {
        "machine": machine(),
        "config": {
            "n_logs": n_logs,
            "warmup": WARMUP,
            "bootstrap": BOOTSTRAP,
            "seeds": list(SEEDS),
            "pool_size": POOL_SIZE,
            "workload_digest": workload.digest,
            "environment_digest": environment.digest,
            "nominal_coverage": NOMINAL_COVERAGE,
            "adaptive_block_size": BLOCK_SIZE,
            "adaptive_seeds": list(ADAPTIVE_SEEDS),
            "adaptive_n_logs": ADAPTIVE_LOGS,
        },
        "results": measurements,
        "tables": {
            "accuracy": rows,
            "dr_detail": detail_rows,
            "screening": [
                {"sweep": "vs a converged incumbent", **rejection.summary_row()},
                {"sweep": "vs a stale cost-blind incumbent", **admission.summary_row()},
            ],
            "adaptive_coverage": adaptive_table,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="off-policy evaluation accuracy benchmark")
    add_common_arguments(parser)
    parser.add_argument("--logs", type=int, default=DEFAULT_LOGS)
    args = parser.parse_args(argv)

    payload = run(n_logs=args.logs)
    write_results("ope.json", payload, reference=args.update_reference)

    print(f"# estimator accuracy against the simulator's exact values ({args.logs:,} logged rows)")
    print(f"workload digest {payload['config']['workload_digest'][:16]}")
    print()
    print(markdown_table(payload["tables"]["accuracy"]))
    print("\n# doubly robust, per candidate (first seed)")
    print(
        markdown_table(
            [
                {k: v for k, v in row.items() if k != "seed"}
                for row in payload["tables"]["dr_detail"]
            ]
        )
    )
    print("\n# what screening did with those estimates")
    print(markdown_table(payload["tables"]["screening"]))
    print(
        "\n# DR interval coverage when the logging policy keeps learning (the gate's own case), "
        f"{ADAPTIVE_LOGS:,} rows"
    )
    print(markdown_table(payload["tables"]["adaptive_coverage"]))
    print(f"\nmeasured in {payload['seconds']}s on {payload['machine']['cpu_count']} CPUs")

    if args.check:
        problems = threshold_problems("bench_ope", payload["results"])
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all OPE benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
