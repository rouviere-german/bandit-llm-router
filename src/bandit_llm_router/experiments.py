"""Simulation runners: the experiments that produce every number this repo claims.

One place defines how a policy is run against the environment, so the bench scripts, the offline
demo and the integration tests all measure the same thing:

* :func:`run_simulation` -- drive one policy over ``n`` requests, logging the action, its
propensity,
  the realized reward, and two expectations per step: the value of the policy's whole distribution
  (``sum_a p(a) E[r|x,a]``) and the oracle's best arm. Regret is computed from those expectations,
  so it measures decisions rather than action-sampling noise.

  Two regret baselines are reported, because they answer different questions. **Oracle regret** is
  measured against the unconstrained best arm; it cannot be sublinear here, since a propensity floor
  spends a fixed fraction of traffic on exploration forever and a linear reward model cannot exactly
  represent a sigmoid quality curve -- both leave a constant per-step gap. **In-class regret** is
  measured against the clairvoyant policy *with the same propensity floor*: the best a policy could
  do while still logging usable propensities. That is the number that isolates learning, and it is
  what the regression thresholds track.
* :func:`true_policy_value` / :func:`expected_policy_cost` -- the simulator knows every arm's
expected
  reward, so a frozen policy's value and cost are computed exactly instead of estimated. That exact
  value is the ground truth the off-policy estimators are scored against.
* :func:`run_ope_accuracy` -- log with a frozen policy, evaluate candidates with IPS/SNIPS/DR, and
  report bias, RMSE and empirical interval coverage against the exact values.
* :func:`run_shift_experiment` -- how many requests a policy needs to get back to 95% of oracle
  reward after a scheduled quality shift, discounted posteriors versus stationary ones.
* :func:`run_screening_trials` -- how often OPE screening rejects candidates that are genuinely
worse
  and admits ones that are genuinely better.
* :class:`PoisonedFeedbackChannel` -- the adversarial-feedback ablation: the same poisoned stream
with
  no defenses, with clipping only, and with the full robust aggregation.

The logging phase of the OPE accuracy experiments runs a *frozen* policy on purpose. An online
learner's log is not i.i.d., so bootstrap intervals over its rows would be measuring something they
do not claim to measure. Freezing costs a short warm-up and makes the reported coverage meaningful.

The shipped governance gate does not get that luxury: ``RouterService.screen`` runs on the log the
live policy wrote while it was learning. :func:`run_adaptive_coverage` measures what that costs --
the same candidates and seeds, logged once by a frozen policy and once by a learner, with the DR
interval computed both by the row bootstrap and by a block bootstrap over contiguous rows. The
degradation is a number in ``bench/results/ope.json``, not a caveat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from bandit_llm_router.feedback import FeedbackPolicy, SourceReport, clip_reward, robust_aggregate
from bandit_llm_router.ope.estimators import Estimate, LoggedBatch, evaluate_policy
from bandit_llm_router.ope.screening import ScreeningConfig, ScreeningReport, screen_candidate
from bandit_llm_router.policies.base import sample_action_from_uniform
from bandit_llm_router.simulator.environment import SimulatedEnvironment

__all__ = [
    "AdaptiveCoverage",
    "AggregationMode",
    "CandidateFactory",
    "DirectFeedbackChannel",
    "LinearBanditResult",
    "OPEAccuracy",
    "PoisonedFeedbackChannel",
    "PolicyFactory",
    "RewardChannel",
    "ScreeningTrials",
    "ShiftRecovery",
    "SimulationLog",
    "SimulationResult",
    "expected_policy_cost",
    "regret_slope",
    "run_adaptive_coverage",
    "run_linear_bandit",
    "run_ope_accuracy",
    "run_screening_trials",
    "run_shift_experiment",
    "run_simulation",
    "true_policy_value",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
AggregationMode = Literal["none", "clip_only", "robust", "first_report_only"]


class SimulationPolicy(Protocol):
    """What a simulation needs from a policy."""

    name: str

    @property
    def version(self) -> str: ...

    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray: ...

    def probabilities_matrix(
        self, features: FloatArray, rng: np.random.Generator
    ) -> FloatArray: ...

    def update(
        self, features: FloatArray, arm_index: int, reward: float, weight: float = 1.0
    ) -> None: ...


PolicyFactory = Callable[[], SimulationPolicy]
CandidateFactory = Callable[[SimulatedEnvironment, int], SimulationPolicy]


class RewardChannel(Protocol):
    """Turns a realized reward into what the learner gets to see, plus a learning weight."""

    def observe(
        self, step: int, true_reward: float, rng: np.random.Generator
    ) -> tuple[float, float]: ...


@dataclass(frozen=True, slots=True)
class DirectFeedbackChannel:
    """The trusted case: the learner sees the realized reward with full weight."""

    def observe(
        self, step: int, true_reward: float, rng: np.random.Generator
    ) -> tuple[float, float]:
        del step, rng
        return true_reward, 1.0


@dataclass(frozen=True, slots=True)
class PoisonedFeedbackChannel:
    """Honest sources plus an attacker that inverts the signal on a fraction of decisions.

    ``mode`` selects the defense under test:

    * ``"none"`` -- the naive implementation: the mean of whatever was reported, unclipped.
    * ``"clip_only"`` -- values clipped into the reward range, then averaged.
    * ``"robust"`` -- the full aggregation pipeline on *every* report for a decision: clipping,
      per-source medians, a weight-capped weighted median across sources, and a reduced learning
      weight for uncorroborated rewards. This is what the **stored** reward looks like -- the one
      off-policy evaluation and offline retraining read -- and it is what
      :func:`bandit_llm_router.feedback.robust_aggregate` computes.
    * ``"first_report_only"`` -- what the **online posterior** actually receives.
      ``RouterService.submit_feedback`` teaches the live policy on the first accepted report for a
      decision and never again (an online ridge update cannot be withdrawn, so applying several
      would double-count one outcome). With one report there is one source, so the per-source median
      and the weighted median have nothing to combine: the online path's only defenses are clipping
      and the reduced ``single_source_weight``. ``attacker_reports_first`` decides who wins the race
      on an attacked decision; it defaults to the attacker, because report arrival order is not
      something the router controls and an attacker that wants to be first can be.

    ``"robust"`` and ``"first_report_only"`` are therefore both real and they measure different
    things. Quoting only the first as "the shipped defense" overstates what the deployed online
    learner is protected by, which is why ``bench/bench_router.py`` publishes both.
    """

    attack_rate: float
    mode: AggregationMode = "robust"
    honest_sources: int = 2
    honest_noise: float = 0.05
    attack_magnitude: float = 50.0
    attacker_reports_first: bool = True
    policy: FeedbackPolicy = field(default_factory=FeedbackPolicy)

    def __post_init__(self) -> None:
        if not 0.0 <= self.attack_rate <= 1.0:
            raise ValueError("attack_rate must be in [0, 1]")
        if self.honest_sources < 1:
            raise ValueError("at least one honest source is required")

    def observe(
        self, step: int, true_reward: float, rng: np.random.Generator
    ) -> tuple[float, float]:
        del step
        reports = [
            SourceReport(f"rubric_{index}", true_reward + float(rng.normal(0.0, self.honest_noise)))
            for index in range(self.honest_sources)
        ]
        attacked = rng.random() < self.attack_rate
        if attacked:
            hostile = -self.attack_magnitude if true_reward >= 0.5 else self.attack_magnitude
            reports.append(SourceReport("attacker", hostile))
        if self.mode == "first_report_only":
            first = reports[-1] if attacked and self.attacker_reports_first else reports[0]
            clipped_value, _ = clip_reward(first.reward, self.policy)
            return clipped_value, self.policy.single_source_weight
        if self.mode == "robust":
            clipped = [
                SourceReport(item.source, clip_reward(item.reward, self.policy)[0])
                for item in reports
            ]
            aggregated = robust_aggregate(clipped, self.policy)
            return aggregated.reward, aggregated.weight
        if self.mode == "clip_only":
            values = [clip_reward(item.reward, self.policy)[0] for item in reports]
        else:
            values = [item.reward for item in reports]
        return float(np.mean(values)), 1.0


@dataclass(frozen=True, slots=True)
class SimulationLog:
    """Per-step arrays from one simulation run."""

    pool_indices: IntArray
    actions: IntArray
    propensities: FloatArray
    rewards: FloatArray
    learner_rewards: FloatArray
    weights: FloatArray
    qualities: FloatArray
    costs: FloatArray
    expected_rewards: FloatArray
    oracle_rewards: FloatArray
    decision_values: FloatArray
    class_references: FloatArray

    @property
    def n(self) -> int:
        return int(self.actions.shape[0])

    def to_batch(self, environment: SimulatedEnvironment) -> LoggedBatch:
        """Materialize the feature rows and wrap the log as an OPE batch."""
        return LoggedBatch(
            features=environment.feature_matrix[self.pool_indices],
            actions=self.actions,
            propensities=self.propensities,
            rewards=self.rewards,
            n_arms=environment.n_arms,
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Headline numbers for one policy over one run."""

    policy_name: str
    policy_version: str
    n: int
    mean_reward: float
    mean_expected_reward: float
    mean_quality: float
    mean_cost_micros: float
    total_cost_usd: float
    arm_shares: tuple[float, ...]
    min_propensity: float
    final_regret: float
    regret_slope: float
    final_class_regret: float
    class_regret_slope: float
    cumulative_regret: FloatArray
    cumulative_class_regret: FloatArray
    log: SimulationLog

    def summary_row(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "mean_reward": round(self.mean_reward, 4),
            "mean_quality": round(self.mean_quality, 4),
            "cost_usd_per_1k": round(self.mean_cost_micros / 1_000, 3),
            "oracle_regret": round(self.final_regret, 1),
            "oracle_slope": round(self.regret_slope, 3),
            "class_regret": round(self.final_class_regret, 1),
            "class_slope": round(self.class_regret_slope, 3),
            "arm_shares": tuple(round(share, 3) for share in self.arm_shares),
        }


def regret_slope(cumulative_regret: FloatArray, *, start_fraction: float = 0.1) -> float:
    """Slope of ``log R(t)`` against ``log t`` over the tail of the run.

    A slope near 1 means regret is still growing linearly (no learning); sublinear growth gives a
    slope below 1, and the usual ``sqrt(T)`` rate shows up as roughly 0.5. Fitted on the tail so the
    initial exploration burst does not dominate the fit.
    """
    from scipy import stats  # noqa: PLC0415 - imported lazily; scipy is slow to import

    values = np.asarray(cumulative_regret, dtype=np.float64)
    if values.ndim != 1 or values.size < 20:
        raise ValueError("need at least 20 cumulative-regret points")
    if not 0.0 < start_fraction < 1.0:
        raise ValueError("start_fraction must be in (0, 1)")
    start = max(5, int(values.size * start_fraction))
    steps = np.arange(1, values.size + 1, dtype=np.float64)[start:]
    tail = values[start:]
    positive = tail > 0.0
    if positive.sum() < 10:
        return 0.0
    result = stats.linregress(np.log(steps[positive]), np.log(tail[positive]))
    return float(result.slope)


def run_simulation(  # noqa: PLR0915 - the per-step loop is deliberately flat and inlined
    environment: SimulatedEnvironment,
    policy: SimulationPolicy,
    *,
    n: int,
    seed: int,
    learn: bool = True,
    channel: RewardChannel | None = None,
    stream: IntArray | None = None,
) -> SimulationResult:
    """Run ``policy`` over ``n`` simulated requests.

    ``stream`` replays a fixed sequence of pool indices so several policies can be compared on
    exactly the same requests. ``channel`` injects a feedback pipeline between the realized reward
    and the learner (see :class:`PoisonedFeedbackChannel`).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed ^ 0x9E3779B9)
    channel_rng = np.random.default_rng(seed + 977)
    indices = (
        np.asarray(stream, dtype=np.int64)
        if stream is not None
        else environment.workload.sample_indices(n, rng)
    )
    if indices.shape != (n,):
        raise ValueError(f"stream must have shape ({n},)")
    features = environment.feature_matrix
    costs = environment.cost_matrix
    noise = environment.draw_noise(n, rng)
    uniforms = rng.random(n)
    noise_sd = environment.noise_sd
    cost_weight = environment.reward_config.cost_weight
    cost_reference = float(environment.reward_config.cost_reference_micros)

    actions = np.empty(n, dtype=np.int64)
    propensities = np.empty(n, dtype=np.float64)
    rewards = np.empty(n, dtype=np.float64)
    learner_rewards = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    qualities = np.empty(n, dtype=np.float64)
    realized_costs = np.empty(n, dtype=np.float64)
    expected = np.empty(n, dtype=np.float64)
    oracle = np.empty(n, dtype=np.float64)
    decision_values = np.empty(n, dtype=np.float64)
    class_reference = np.empty(n, dtype=np.float64)
    floor = float(getattr(policy, "propensity_floor", 0.0))

    regimes = environment.regimes
    boundaries = [*(regime.start_step for regime in regimes[1:]), n]
    for regime, stop_at in zip(regimes, boundaries, strict=True):
        start, stop = regime.start_step, min(n, stop_at)
        if start >= n:
            break
        expected_quality = regime.expected_quality
        expected_reward = regime.expected_reward
        best_reward = regime.best_reward
        # The in-class reference: the best policy the learner's own model class contains, run with
        # the same propensity floor. Both constraints matter -- the floor is a deliberate design
        # choice and linearity is the model family, so charging a learner for either would make its
        # regret grow linearly no matter how well it learns.
        floored_best = (
            1.0 - floor * environment.n_arms
        ) * regime.linear_best_reward + floor * expected_reward.sum(axis=1)
        for step in range(start, stop):
            index = int(indices[step])
            context = features[index]
            distribution = policy.probabilities(context, policy_rng)
            arm = sample_action_from_uniform(distribution, float(uniforms[step]))
            quality = expected_quality[index, arm] + noise[step] * noise_sd[arm]
            quality = 0.0 if quality < 0.0 else (1.0 if quality > 1.0 else quality)
            cost = costs[index, arm]
            penalty = cost_weight * min(1.0, cost / cost_reference)
            reward = quality - penalty
            reward = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)
            if channel is None:
                observed, weight = reward, 1.0
            else:
                observed, weight = channel.observe(step, reward, channel_rng)
            if learn:
                policy.update(context, arm, observed, weight)
            actions[step] = arm
            propensities[step] = distribution[arm]
            rewards[step] = reward
            learner_rewards[step] = observed
            weights[step] = weight
            qualities[step] = quality
            realized_costs[step] = cost
            expected[step] = expected_reward[index, arm]
            oracle[step] = best_reward[index]
            decision_values[step] = float(distribution @ expected_reward[index])
            class_reference[step] = floored_best[index]

    cumulative_regret = np.cumsum(oracle - decision_values)
    cumulative_class_regret = np.cumsum(np.maximum(0.0, class_reference - decision_values))
    log = SimulationLog(
        pool_indices=indices,
        actions=actions,
        propensities=propensities,
        rewards=rewards,
        learner_rewards=learner_rewards,
        weights=weights,
        qualities=qualities,
        costs=realized_costs,
        expected_rewards=expected,
        oracle_rewards=oracle,
        decision_values=decision_values,
        class_references=class_reference,
    )
    shares = np.bincount(actions, minlength=environment.n_arms) / n
    return SimulationResult(
        policy_name=policy.name,
        policy_version=policy.version,
        n=n,
        mean_reward=float(rewards.mean()),
        mean_expected_reward=float(expected.mean()),
        mean_quality=float(qualities.mean()),
        mean_cost_micros=float(realized_costs.mean()),
        total_cost_usd=float(realized_costs.sum() / 1_000_000.0),
        arm_shares=tuple(float(share) for share in shares),
        min_propensity=float(propensities.min()),
        final_regret=float(cumulative_regret[-1]),
        regret_slope=regret_slope(cumulative_regret) if n >= 200 else 0.0,
        final_class_regret=float(cumulative_class_regret[-1]),
        class_regret_slope=regret_slope(cumulative_class_regret) if n >= 200 else 0.0,
        cumulative_regret=cumulative_regret,
        cumulative_class_regret=cumulative_class_regret,
        log=log,
    )


def true_policy_value(
    environment: SimulatedEnvironment,
    policy: SimulationPolicy,
    stream: IntArray,
    *,
    rng: np.random.Generator,
    step: int = 0,
) -> float:
    """Exact per-request value of a frozen ``policy`` on ``stream`` (no sampling noise)."""
    regime = environment.regime_at(step)
    probabilities = policy.probabilities_matrix(environment.feature_matrix, rng)
    per_request = np.einsum("ij,ij->i", probabilities, regime.expected_reward)
    return float(per_request[np.asarray(stream, dtype=np.int64)].mean())


def expected_policy_cost(
    environment: SimulatedEnvironment,
    policy: SimulationPolicy,
    stream: IntArray,
    *,
    rng: np.random.Generator,
) -> float:
    """Exact expected cost per request in micro-USD for a frozen ``policy`` on ``stream``."""
    probabilities = policy.probabilities_matrix(environment.feature_matrix, rng)
    per_request = np.einsum("ij,ij->i", probabilities, environment.cost_matrix)
    return float(per_request[np.asarray(stream, dtype=np.int64)].mean())


@dataclass(frozen=True, slots=True)
class OPEAccuracy:
    """Accuracy of one estimator over a set of candidates and seeds."""

    method: str
    bias: float
    rmse: float
    coverage: float
    mean_ess: float
    trials: int

    def summary_row(self) -> dict[str, object]:
        return {
            "estimator": self.method,
            "bias": round(self.bias, 4),
            "rmse": round(self.rmse, 4),
            "coverage_95": round(self.coverage, 3),
            "mean_ess": round(self.mean_ess, 1),
            "trials": self.trials,
        }


def _logging_run(
    environment: SimulatedEnvironment,
    logger_factory: PolicyFactory,
    *,
    warmup: int,
    n_logs: int,
    seed: int,
    learn_while_logging: bool = False,
) -> tuple[SimulationPolicy, SimulationResult]:
    """Warm a logging policy up, then log ``n_logs`` decisions with it.

    ``learn_while_logging`` is the difference between the two regimes this repo cares about. False
    freezes the policy first, so the logged rows are i.i.d. given the context stream and a bootstrap
    over rows means what it says. True keeps updating, which is what the deployed service's own
    decision log looks like: every row's propensity depends on the rewards of the rows before it.
    """
    logger = logger_factory()
    if warmup > 0:
        run_simulation(environment, logger, n=warmup, seed=seed, learn=True)
    logging_run = run_simulation(
        environment, logger, n=n_logs, seed=seed + 1, learn=learn_while_logging
    )
    return logger, logging_run


def run_ope_accuracy(
    environment: SimulatedEnvironment,
    *,
    logger_factory: PolicyFactory,
    candidates: Mapping[str, CandidateFactory],
    n_logs: int,
    seeds: Sequence[int],
    warmup: int = 2_000,
    bootstrap: int = 200,
    clip: float | None = None,
) -> tuple[dict[str, OPEAccuracy], list[dict[str, object]]]:
    """Score IPS, SNIPS and DR against the simulator's exact policy values.

    Returns per-estimator accuracy plus one detail row per (seed, candidate) for inspection.
    """
    errors: dict[str, list[float]] = {"ips": [], "snips": [], "dr": []}
    covered: dict[str, list[float]] = {"ips": [], "snips": [], "dr": []}
    ess: dict[str, list[float]] = {"ips": [], "snips": [], "dr": []}
    details: list[dict[str, object]] = []
    for seed in seeds:
        _, logging_run = _logging_run(
            environment, logger_factory, warmup=warmup, n_logs=n_logs, seed=seed
        )
        batch = logging_run.log.to_batch(environment)
        for name, factory in candidates.items():
            candidate = factory(environment, seed)
            evaluation_rng = np.random.default_rng(seed + 31)
            probabilities = candidate.probabilities_matrix(batch.features, evaluation_rng)
            truth = true_policy_value(
                environment,
                candidate,
                logging_run.log.pool_indices,
                rng=np.random.default_rng(seed + 31),
            )
            report = evaluate_policy(
                batch,
                probabilities,
                bootstrap=bootstrap,
                clip=clip,
                rng=np.random.default_rng(seed + 57),
            )
            for method, estimate in (
                ("ips", report.ips),
                ("snips", report.snips),
                ("dr", report.dr),
            ):
                errors[method].append(estimate.value - truth)
                covered[method].append(1.0 if estimate.ci_low <= truth <= estimate.ci_high else 0.0)
                ess[method].append(estimate.ess)
                details.append(
                    {
                        "seed": seed,
                        "candidate": name,
                        "estimator": method,
                        "truth": round(truth, 4),
                        "estimate": round(estimate.value, 4),
                        "ci_low": round(estimate.ci_low, 4),
                        "ci_high": round(estimate.ci_high, 4),
                        "ess": round(estimate.ess, 1),
                    }
                )
    accuracy = {
        method: OPEAccuracy(
            method=method,
            bias=float(np.mean(errors[method])),
            rmse=float(np.sqrt(np.mean(np.square(errors[method])))),
            coverage=float(np.mean(covered[method])),
            mean_ess=float(np.mean(ess[method])),
            trials=len(errors[method]),
        )
        for method in errors
    }
    return accuracy, details


def _arm_share_drift(actions: IntArray, n_arms: int) -> float:
    """Total-variation distance between the first and last tenth of a log's arm shares.

    How much the logging policy moved *while it was logging*. It is the non-vacuity check on any
    claim about adaptively collected data: if this is near zero the "online" log is a frozen log
    wearing a different label, and comparing bootstraps on it proves nothing.
    """
    rows = int(actions.shape[0])
    tenth = max(1, rows // 10)
    first = np.bincount(actions[:tenth], minlength=n_arms) / tenth
    last = np.bincount(actions[-tenth:], minlength=n_arms) / tenth
    return 0.5 * float(np.abs(first - last).sum())


@dataclass(frozen=True, slots=True)
class AdaptiveCoverage:
    """DR interval quality for one (logging regime, bootstrap) pair, scored against exact values."""

    logging_regime: str
    bootstrap_block_size: int | None
    coverage: float
    mean_interval_width: float
    bias: float
    rmse: float
    trials: int
    logging_drift: float
    n_logs: int

    def summary_row(self) -> dict[str, object]:
        block = self.bootstrap_block_size
        return {
            "logging policy": self.logging_regime,
            "logged rows": self.n_logs,
            "bootstrap": "rows" if block is None else f"blocks of {block}",
            "dr coverage_95": round(self.coverage, 3),
            "mean CI width": round(self.mean_interval_width, 4),
            "bias": round(self.bias, 4),
            "rmse": round(self.rmse, 4),
            "arm-share drift": round(self.logging_drift, 3),
            "trials": self.trials,
        }


def run_adaptive_coverage(
    environment: SimulatedEnvironment,
    *,
    logger_factory: PolicyFactory,
    candidates: Mapping[str, CandidateFactory],
    n_logs: int,
    seeds: Sequence[int],
    warmup: int = 2_000,
    bootstrap: int = 200,
    block_sizes: Sequence[int | None] = (None,),
    learn_while_logging: bool = True,
    regime_label: str | None = None,
) -> list[AdaptiveCoverage]:
    """What the DR interval is worth when the log was written by a policy that kept learning.

    This is the shipped governance path, not an academic aside: ``RouterService.screen`` evaluates
    candidates on the live policy's own decision log, and that policy updates on the first accepted
    reward report for every decision. The rows are therefore serially dependent, and a percentile
    bootstrap over individual rows is assuming they are not.

    One logging run per seed is shared by every entry in ``block_sizes``, so the row bootstrap and
    the block bootstrap are compared on *identical* data and the difference can only be the
    resampling scheme. Returns one :class:`AdaptiveCoverage` per block size.
    """
    if not block_sizes:
        raise ValueError("at least one block size is required")
    label = regime_label or ("online learner" if learn_while_logging else "frozen")
    errors: dict[int, list[float]] = {index: [] for index in range(len(block_sizes))}
    covered: dict[int, list[float]] = {index: [] for index in range(len(block_sizes))}
    widths: dict[int, list[float]] = {index: [] for index in range(len(block_sizes))}
    drifts: list[float] = []
    for seed in seeds:
        _, logging_run = _logging_run(
            environment,
            logger_factory,
            warmup=warmup,
            n_logs=n_logs,
            seed=seed,
            learn_while_logging=learn_while_logging,
        )
        batch = logging_run.log.to_batch(environment)
        drifts.append(_arm_share_drift(logging_run.log.actions, environment.n_arms))
        for name, factory in candidates.items():
            del name
            candidate = factory(environment, seed)
            probabilities = candidate.probabilities_matrix(
                batch.features, np.random.default_rng(seed + 31)
            )
            truth = true_policy_value(
                environment,
                candidate,
                logging_run.log.pool_indices,
                rng=np.random.default_rng(seed + 31),
            )
            for index, block_size in enumerate(block_sizes):
                estimate = evaluate_policy(
                    batch,
                    probabilities,
                    bootstrap=bootstrap,
                    rng=np.random.default_rng(seed + 57),
                    block_size=block_size,
                ).dr
                errors[index].append(estimate.value - truth)
                covered[index].append(1.0 if estimate.ci_low <= truth <= estimate.ci_high else 0.0)
                widths[index].append(estimate.ci_high - estimate.ci_low)
    return [
        AdaptiveCoverage(
            logging_regime=label,
            bootstrap_block_size=block_size,
            coverage=float(np.mean(covered[index])),
            mean_interval_width=float(np.mean(widths[index])),
            bias=float(np.mean(errors[index])),
            rmse=float(np.sqrt(np.mean(np.square(errors[index])))),
            trials=len(errors[index]),
            logging_drift=float(np.mean(drifts)),
            n_logs=n_logs,
        )
        for index, block_size in enumerate(block_sizes)
    ]


@dataclass(frozen=True, slots=True)
class ShiftRecovery:
    """How a policy handled a scheduled quality shift.

    ``recovery_requests`` is ``0`` both when a policy bounced back within one step and when the
    shift never pushed it below the target at all, so ``dipped_below_target`` separates the two.
    They mean
    very different things -- the second is the best possible outcome and printing it as a bare "0"
    invites reading it as a measurement failure.
    """

    policy_name: str
    discount: float
    pre_shift_ratio: float
    post_shift_trough: float
    recovery_requests: int | None
    final_ratio: float
    dipped_below_target: bool

    def summary_row(self) -> dict[str, object]:
        if not self.dipped_below_target:
            recovery: object = "never fell below target"
        elif self.recovery_requests is None:
            recovery = "not recovered"
        else:
            recovery = self.recovery_requests
        return {
            "policy": self.policy_name,
            "discount": self.discount,
            "pre_shift_ratio": round(self.pre_shift_ratio, 4),
            "trough_ratio": round(self.post_shift_trough, 4),
            "requests_to_recover": recovery,
            "final_ratio": round(self.final_ratio, 4),
        }


def _moving_ratio(numerator: FloatArray, denominator: FloatArray, window: int) -> FloatArray:
    """Trailing moving average of ``numerator / denominator`` over ``window`` steps."""
    cumulative_numerator = np.concatenate([[0.0], np.cumsum(numerator)])
    cumulative_denominator = np.concatenate([[0.0], np.cumsum(denominator)])
    size = numerator.shape[0]
    starts = np.maximum(0, np.arange(1, size + 1) - window)
    top = cumulative_numerator[1:] - cumulative_numerator[starts]
    bottom = cumulative_denominator[1:] - cumulative_denominator[starts]
    return np.where(bottom > 0.0, top / np.maximum(bottom, 1e-12), 0.0)


def run_shift_experiment(
    environment: SimulatedEnvironment,
    *,
    policies: Mapping[str, tuple[float, SimulationPolicy]],
    n: int,
    seed: int,
    shift_step: int,
    window: int = 2_000,
    target: float = 0.95,
) -> list[ShiftRecovery]:
    """Run each policy over the same stream and measure recovery after ``shift_step``.

    ``policies`` maps a label to ``(discount, policy)``. Three details decide whether the number
    this returns means anything:

    * **The denominator is the in-class reference, not the raw oracle.** A propensity floor spends a
      fixed fraction of traffic on exploration forever, so a converged policy's ratio against the
      unconstrained oracle sits *below* 1 by roughly the floor's own cost -- and a target of 0.95
      would then be met or missed by the floor rather than by adaptation. The reference here is the
      clairvoyant policy with the *same* floor, so 1.0 is achievable and "95% recovered" is a
      statement about learning.
    * **The ratio is a trailing average**, so it needs ``window`` steps to reflect the new regime.
      The trough is therefore searched over the first ``2 * window`` post-shift steps rather than
      the first ``window``.
    * **Recovery is only counted after an actual dip.** Immediately after the shift the trailing
      window is still mostly pre-shift data, so the ratio trivially clears the target; searching
      from the shift would report 0 requests for every policy, including one that never adapts.
      Recovery is measured from the first step where the ratio fell *below* the target, and is
      ``None`` when it never came back.

    Decision values are used throughout rather than realized rewards, so the curve is not blurred by
    reward noise or action sampling.
    """
    if not 0 < shift_step < n:
        raise ValueError("shift_step must be inside the run")
    stream = environment.workload.sample_indices(n, np.random.default_rng(seed))
    out: list[ShiftRecovery] = []
    for label, (discount, policy) in policies.items():
        result = run_simulation(environment, policy, n=n, seed=seed, stream=stream)
        ratio = _moving_ratio(result.log.decision_values, result.log.class_references, window)
        pre = float(ratio[shift_step - 1])
        post = ratio[shift_step:]
        trough = float(post[: min(2 * window, post.size)].min())
        below = np.nonzero(post < target)[0]
        dipped = below.size > 0
        if not dipped:
            recovery: int | None = 0
        else:
            first_dip = int(below[0])
            regained = np.nonzero(post[first_dip:] >= target)[0]
            recovery = first_dip + int(regained[0]) if regained.size else None
        out.append(
            ShiftRecovery(
                policy_name=label,
                discount=discount,
                pre_shift_ratio=pre,
                post_shift_trough=trough,
                recovery_requests=recovery,
                final_ratio=float(ratio[-1]),
                dipped_below_target=dipped,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class ScreeningTrials:
    """How screening behaved over many (seed, candidate) trials."""

    truly_worse: int
    truly_worse_rejected: int
    truly_better: int
    truly_better_admitted: int
    reports: tuple[ScreeningReport, ...]

    @property
    def rejection_rate(self) -> float:
        return self.truly_worse_rejected / self.truly_worse if self.truly_worse else 0.0

    @property
    def admission_rate(self) -> float:
        return self.truly_better_admitted / self.truly_better if self.truly_better else 0.0

    def summary_row(self) -> dict[str, object]:
        return {
            "truly_worse_candidates": self.truly_worse,
            "rejected": self.truly_worse_rejected,
            "rejection_rate": round(self.rejection_rate, 3),
            "truly_better_candidates": self.truly_better,
            "admitted": self.truly_better_admitted,
            "admission_rate": round(self.admission_rate, 3),
        }


def run_screening_trials(
    environment: SimulatedEnvironment,
    *,
    logger_factory: PolicyFactory,
    candidates: Mapping[str, CandidateFactory],
    n_logs: int,
    seeds: Sequence[int],
    warmup: int = 2_000,
    config: ScreeningConfig | None = None,
    incumbent_factory: CandidateFactory | None = None,
) -> ScreeningTrials:
    """Screen every candidate on freshly logged data and compare the verdict with the truth.

    ``incumbent_factory`` separates *who wrote the log* from *what is live now*, which is how the
    service works: the log was produced by whatever policy was serving, and the candidate is
    screened against the policy deployed now. Left unset, the logging policy is also the
    incumbent -- convenient, but it makes the admission side of the measurement vacuous whenever the
    logger has already converged, because then no candidate can be meaningfully better than it.
    """
    settings = config if config is not None else ScreeningConfig()
    worse = worse_rejected = better = better_admitted = 0
    reports: list[ScreeningReport] = []
    for seed in seeds:
        logger, logging_run = _logging_run(
            environment, logger_factory, warmup=warmup, n_logs=n_logs, seed=seed
        )
        incumbent = incumbent_factory(environment, seed) if incumbent_factory else logger
        batch = logging_run.log.to_batch(environment)
        expected_cost = environment.cost_matrix[logging_run.log.pool_indices]
        ceiling_cost = environment.ceiling_matrix[logging_run.log.pool_indices]
        incumbent_truth = true_policy_value(
            environment,
            incumbent,
            logging_run.log.pool_indices,
            rng=np.random.default_rng(seed + 11),
        )
        for factory in candidates.values():
            candidate = factory(environment, seed)
            truth = true_policy_value(
                environment,
                candidate,
                logging_run.log.pool_indices,
                rng=np.random.default_rng(seed + 13),
            )
            report = screen_candidate(
                batch,
                candidate=candidate,
                incumbent=incumbent,
                expected_cost_micros=expected_cost,
                ceiling_cost_micros=ceiling_cost,
                config=settings,
                rng=np.random.default_rng(seed + 17),
            )
            reports.append(report)
            if truth < incumbent_truth:
                worse += 1
                worse_rejected += 0 if report.eligible else 1
            else:
                better += 1
                better_admitted += 1 if report.eligible else 0
    return ScreeningTrials(
        truly_worse=worse,
        truly_worse_rejected=worse_rejected,
        truly_better=better,
        truly_better_admitted=better_admitted,
        reports=tuple(reports),
    )


@dataclass(frozen=True, slots=True)
class LinearBanditResult:
    """Regret of a policy on a well-specified linear bandit."""

    policy_name: str
    n: int
    final_regret: float
    regret_slope: float
    cumulative_regret: FloatArray

    def summary_row(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "n": self.n,
            "regret": round(self.final_regret, 1),
            "regret_slope": round(self.regret_slope, 3),
        }


def run_linear_bandit(
    policy: SimulationPolicy,
    *,
    features: FloatArray,
    thetas: FloatArray,
    n: int,
    seed: int,
    noise_sd: float = 0.1,
) -> LinearBanditResult:
    """Run a policy on a bandit whose expected reward really is linear in the features.

    ``E[r | x, a] = clip(x . theta_a, 0, 1)``. This is the setting LinUCB and Thompson sampling are
    analysed in, so it is the only place where a sublinear regret *rate* is a claim about the
    implementation rather than about the simulator's realism. The production simulator deliberately
    violates realizability (sigmoid quality curves over a latent difficulty), and its regret is
    reported separately.

    Regret is measured against the clairvoyant policy *with the same propensity floor*, as in
    :func:`run_simulation`: the floor's fixed exploration cost is a deliberate design choice, not a
    learning failure, and leaving it in would make every curve linear by construction.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if thetas.ndim != 2 or thetas.shape[1] != features.shape[1]:
        raise ValueError("thetas must have shape (n_arms, dim)")
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed ^ 0x5DEECE66)
    pool = features.shape[0]
    indices = rng.integers(0, pool, size=n)
    noise = rng.normal(0.0, noise_sd, size=n)
    uniforms = rng.random(n)
    values = np.clip(features @ thetas.T, 0.0, 1.0)
    floor = float(getattr(policy, "propensity_floor", 0.0))
    n_arms = values.shape[1]
    reference = (1.0 - floor * n_arms) * values.max(axis=1) + floor * values.sum(axis=1)
    regret = np.empty(n, dtype=np.float64)
    for step in range(n):
        index = int(indices[step])
        context = features[index]
        distribution = policy.probabilities(context, policy_rng)
        arm = sample_action_from_uniform(distribution, float(uniforms[step]))
        observed = float(np.clip(values[index, arm] + noise[step], 0.0, 1.0))
        policy.update(context, arm, observed)
        regret[step] = max(0.0, reference[index] - float(distribution @ values[index]))
    cumulative = np.cumsum(regret)
    return LinearBanditResult(
        policy_name=policy.name,
        n=n,
        final_regret=float(cumulative[-1]),
        regret_slope=regret_slope(cumulative) if n >= 200 else 0.0,
        cumulative_regret=cumulative,
    )


def estimate_table(estimates: Sequence[Estimate]) -> list[dict[str, object]]:
    """Rows for printing a set of estimates as a table."""
    return [
        {
            "estimator": estimate.method,
            "value": round(estimate.value, 4),
            "ci_low": round(estimate.ci_low, 4),
            "ci_high": round(estimate.ci_high, 4),
            "ess": round(estimate.ess, 1),
        }
        for estimate in estimates
    ]
