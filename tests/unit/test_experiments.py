"""The simulation runners that produce every number the README reports.

The benches call exactly these functions, so what is tested here is what the README claims:

* a learning policy's regret grows sublinearly on a *well-specified* linear bandit -- the only
  setting where a rate is a claim about the implementation rather than about the simulator;
* a learner beats the static baselines on the production simulator;
* the OPE estimators are scored against the simulator's exact policy value, not against each other;
* discounted posteriors recover from a scheduled shift faster than stationary ones;
* screening rejects candidates that are genuinely worse;
* the poisoning defenses are worth something, measured with them on and off.

Everything runs at a fraction of the bench sizes, chosen so the direction of each effect is already
unambiguous. The exact magnitudes live in ``bench/`` with regression thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from bandit_llm_router.experiments import (
    DirectFeedbackChannel,
    PoisonedFeedbackChannel,
    expected_policy_cost,
    regret_slope,
    run_linear_bandit,
    run_ope_accuracy,
    run_screening_trials,
    run_shift_experiment,
    run_simulation,
    true_policy_value,
)
from bandit_llm_router.ope.screening import ScreeningConfig
from bandit_llm_router.policies.baselines import KeywordRule, StaticArm
from bandit_llm_router.policies.linear_bandit import (
    EpsilonGreedy,
    LinearBanditPolicy,
    LinearThompson,
    LinUCB,
)
from bandit_llm_router.simulator.environment import SimulatedEnvironment

pytestmark = pytest.mark.unit

FLOOR = 0.05


def learner(
    environment: SimulatedEnvironment, kind: str = "linucb", **overrides: object
) -> LinearBanditPolicy:
    common: dict[str, object] = {
        "arms": environment.arms,
        "dim": environment.feature_dim,
        "feature_version": environment.featurizer.version,
        "propensity_floor": FLOOR,
    }
    if kind == "linucb":
        common.setdefault("alpha", 0.5)
        return LinUCB(**{**common, **overrides})  # type: ignore[arg-type]
    if kind == "thompson":
        common.update({"posterior_sd": 0.3, "mc_samples": 16})
        return LinearThompson(**{**common, **overrides})  # type: ignore[arg-type]
    common.setdefault("epsilon", 0.05)
    return EpsilonGreedy(**{**common, **overrides})  # type: ignore[arg-type]


def static(environment: SimulatedEnvironment, target: str) -> StaticArm:
    return StaticArm(
        arms=environment.arms,
        dim=environment.feature_dim,
        target=target,
        propensity_floor=FLOOR,
        feature_version=environment.featurizer.version,
    )


# --- the regret slope ----------------------------------------------------------------------------


def test_the_slope_separates_linear_growth_from_sublinear() -> None:
    steps = np.arange(1, 5_001, dtype=np.float64)
    assert regret_slope(steps) == pytest.approx(1.0, abs=0.02)
    assert regret_slope(np.sqrt(steps)) == pytest.approx(0.5, abs=0.02)
    assert regret_slope(np.log1p(steps)) < 0.3
    assert regret_slope(np.zeros(5_000)) == 0.0


def test_the_slope_validates_its_input() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        regret_slope(np.arange(5, dtype=np.float64))
    with pytest.raises(ValueError, match="start_fraction"):
        regret_slope(np.arange(100, dtype=np.float64), start_fraction=1.5)


# --- learning on a well-specified linear bandit ---------------------------------------------------


@pytest.mark.parametrize("kind", ["linucb", "thompson"])
def test_regret_is_sublinear_where_the_model_is_correct(kind: str) -> None:
    """The textbook setting: expected reward really is linear, so a rate is a real claim."""
    rng = np.random.default_rng(0)
    dim = 6
    features = np.column_stack([np.ones(400), rng.uniform(-1.0, 1.0, size=(400, dim - 1)) * 0.5])
    thetas = np.array(
        [
            [0.45, 0.30, 0.00, -0.10, 0.05, 0.00],
            [0.50, 0.00, 0.30, 0.05, -0.05, 0.10],
            [0.48, -0.05, 0.05, 0.35, 0.10, 0.00],
        ]
    )
    policy = learner_for_linear(kind, dim)
    result = run_linear_bandit(policy, features=features, thetas=thetas, n=6_000, seed=1)
    assert result.n == 6_000
    assert result.final_regret >= 0.0
    assert result.regret_slope < 0.8
    assert result.summary_row()["policy"] == policy.name


def learner_for_linear(kind: str, dim: int) -> LinearBanditPolicy:
    arms = ("a", "b", "c")
    if kind == "linucb":
        return LinUCB(arms=arms, dim=dim, alpha=0.4, propensity_floor=FLOOR)
    return LinearThompson(
        arms=arms, dim=dim, posterior_sd=0.25, mc_samples=16, propensity_floor=FLOOR
    )


def test_a_non_learning_policy_accumulates_regret_linearly() -> None:
    """The control: without learning the slope stays near 1, so the test above measures learning."""
    rng = np.random.default_rng(2)
    dim = 4
    features = np.column_stack([np.ones(200), rng.uniform(-1.0, 1.0, size=(200, dim - 1))])
    thetas = np.array([[0.2, 0.4, 0.0, 0.0], [0.6, 0.0, 0.3, 0.0], [0.3, 0.0, 0.0, 0.4]])
    frozen = StaticArm(arms=("a", "b", "c"), dim=dim, target="a", propensity_floor=FLOOR)
    result = run_linear_bandit(frozen, features=features, thetas=thetas, n=4_000, seed=3)
    assert result.regret_slope > 0.9


def test_the_linear_bandit_runner_validates_its_inputs() -> None:
    policy = LinUCB(arms=("a", "b"), dim=3)
    features = np.ones((10, 3))
    with pytest.raises(ValueError, match="n must be"):
        run_linear_bandit(policy, features=features, thetas=np.zeros((2, 3)), n=0, seed=0)
    with pytest.raises(ValueError, match="thetas must have shape"):
        run_linear_bandit(policy, features=features, thetas=np.zeros((2, 5)), n=10, seed=0)


# --- learning on the production simulator ---------------------------------------------------------


def test_a_learner_beats_the_static_baselines_on_reward(
    environment: SimulatedEnvironment,
) -> None:
    stream = environment.workload.sample_indices(6_000, np.random.default_rng(9))
    learned = run_simulation(environment, learner(environment), n=6_000, seed=4, stream=stream)
    baselines = {
        target: run_simulation(
            environment, static(environment, target), n=6_000, seed=4, learn=False, stream=stream
        )
        for target in environment.arms
    }
    assert learned.mean_expected_reward > max(
        result.mean_expected_reward for result in baselines.values()
    )
    # And the log it produced is usable: every propensity respects the floor.
    assert learned.min_propensity >= FLOOR - 1e-12


def test_a_learner_is_cheaper_than_always_using_the_largest_model(
    environment: SimulatedEnvironment,
) -> None:
    stream = environment.workload.sample_indices(6_000, np.random.default_rng(10))
    learned = run_simulation(environment, learner(environment), n=6_000, seed=5, stream=stream)
    opus = run_simulation(
        environment, static(environment, "opus"), n=6_000, seed=5, learn=False, stream=stream
    )
    assert learned.mean_cost_micros < opus.mean_cost_micros
    assert learned.total_cost_usd < opus.total_cost_usd


def test_in_class_regret_grows_more_slowly_than_oracle_regret(
    environment: SimulatedEnvironment,
) -> None:
    """The floor's fixed exploration cost is a design choice, so it is reported separately."""
    result = run_simulation(environment, learner(environment), n=6_000, seed=6)
    assert result.final_class_regret < result.final_regret
    assert result.class_regret_slope < result.regret_slope


def test_a_learner_spreads_traffic_across_the_arms(
    environment: SimulatedEnvironment,
) -> None:
    result = run_simulation(environment, learner(environment), n=4_000, seed=7)
    shares = np.asarray(result.arm_shares)
    assert shares.sum() == pytest.approx(1.0)
    assert np.all(shares > FLOOR / 2)
    assert float(shares.max()) < 0.95


def test_a_shared_stream_makes_policies_comparable(
    environment: SimulatedEnvironment,
) -> None:
    stream = environment.workload.sample_indices(1_000, np.random.default_rng(11))
    first = run_simulation(
        environment, static(environment, "haiku"), n=1_000, seed=8, learn=False, stream=stream
    )
    second = run_simulation(
        environment, static(environment, "sonnet"), n=1_000, seed=8, learn=False, stream=stream
    )
    assert np.array_equal(first.log.pool_indices, second.log.pool_indices)
    assert np.array_equal(first.log.qualities == 0.0, first.log.qualities == 0.0)


def test_the_run_is_reproducible(environment: SimulatedEnvironment) -> None:
    first = run_simulation(environment, learner(environment), n=1_500, seed=12)
    second = run_simulation(environment, learner(environment), n=1_500, seed=12)
    assert first.mean_reward == second.mean_reward
    assert np.array_equal(first.log.actions, second.log.actions)
    assert first.final_regret == second.final_regret


def test_a_frozen_policy_does_not_learn(environment: SimulatedEnvironment) -> None:
    policy = learner(environment)
    run_simulation(environment, policy, n=500, seed=13, learn=True)
    updates = policy.updates
    run_simulation(environment, policy, n=500, seed=14, learn=False)
    assert policy.updates == updates


def test_the_simulation_validates_its_arguments(environment: SimulatedEnvironment) -> None:
    policy = learner(environment)
    with pytest.raises(ValueError, match="n must be"):
        run_simulation(environment, policy, n=0, seed=0)
    with pytest.raises(ValueError, match="stream must have shape"):
        run_simulation(environment, policy, n=10, seed=0, stream=np.zeros(5, dtype=np.int64))


def test_the_log_becomes_an_ope_batch(environment: SimulatedEnvironment) -> None:
    result = run_simulation(environment, learner(environment), n=800, seed=15)
    batch = result.log.to_batch(environment)
    assert batch.n == 800
    assert batch.n_arms == environment.n_arms
    assert float(batch.propensities.min()) >= FLOOR - 1e-12
    assert result.log.n == 800
    assert result.summary_row()["policy"] == "linucb"


def test_the_direct_channel_passes_the_reward_through() -> None:
    channel = DirectFeedbackChannel()
    assert channel.observe(0, 0.7, np.random.default_rng(0)) == (0.7, 1.0)


# --- exact policy values --------------------------------------------------------------------------


def test_the_exact_value_of_a_static_policy_is_computable(
    environment: SimulatedEnvironment,
) -> None:
    """The simulator knows every arm's expected reward, so the ground truth needs no estimation."""
    stream = environment.workload.sample_indices(2_000, np.random.default_rng(16))
    rng = np.random.default_rng(17)
    values = {
        target: true_policy_value(environment, static(environment, target), stream, rng=rng)
        for target in environment.arms
    }
    assert all(0.0 < value < 1.0 for value in values.values())
    costs = {
        target: expected_policy_cost(environment, static(environment, target), stream, rng=rng)
        for target in environment.arms
    }
    # Costs rise with the arm size; values do not, which is the trade-off being learned.
    assert costs["haiku"] < costs["sonnet"] < costs["opus"]


def test_the_exact_value_matches_the_simulated_average(
    environment: SimulatedEnvironment,
) -> None:
    stream = environment.workload.sample_indices(20_000, np.random.default_rng(18))
    policy = static(environment, "sonnet")
    exact = true_policy_value(environment, policy, stream, rng=np.random.default_rng(19))
    simulated = run_simulation(environment, policy, n=20_000, seed=20, learn=False, stream=stream)
    assert simulated.mean_reward == pytest.approx(exact, abs=0.01)


# --- OPE accuracy against the truth ---------------------------------------------------------------


def test_the_estimators_are_scored_against_the_simulators_exact_value(
    environment: SimulatedEnvironment,
) -> None:
    accuracy, details = run_ope_accuracy(
        environment,
        logger_factory=lambda: learner(environment, "epsilon_greedy", epsilon=0.2),
        candidates={
            "haiku": lambda env, _seed: static(env, "haiku"),
            "opus": lambda env, _seed: static(env, "opus"),
        },
        n_logs=4_000,
        seeds=(1, 2, 3),
        warmup=800,
        bootstrap=120,
    )
    assert set(accuracy) == {"ips", "snips", "dr"}
    for method, score in accuracy.items():
        assert score.trials == 6, method
        assert score.rmse < 0.25, method
        assert score.mean_ess > 0.0
        assert 0.0 <= score.coverage <= 1.0
    # DR should be the most accurate of the three here: the reward model is decent.
    assert accuracy["dr"].rmse <= accuracy["ips"].rmse
    assert accuracy["dr"].summary_row()["estimator"] == "dr"
    assert len(details) == 18  # one row per (seed, candidate, estimator)


# --- adaptation after a shift ---------------------------------------------------------------------


def test_discounted_posteriors_recover_faster_after_a_quality_shift(
    severe_shift_environment: SimulatedEnvironment,
) -> None:
    """The arm the policy learned to prefer becomes the worst; a stationary posterior is slow."""
    env = severe_shift_environment
    recoveries = run_shift_experiment(
        env,
        policies={
            "stationary": (1.0, learner(env, discount=1.0)),
            "discounted": (0.999, learner(env, discount=0.999)),
        },
        n=12_000,
        seed=21,
        shift_step=2_000,
        window=1_000,
    )
    by_name = {item.policy_name: item for item in recoveries}
    assert set(by_name) == {"stationary", "discounted"}
    for item in recoveries:
        # Both were converged before the shift and both were knocked off it.
        assert item.pre_shift_ratio > 0.9
        assert item.post_shift_trough < item.pre_shift_ratio - 0.05
        assert item.recovery_requests is not None
        assert item.dipped_below_target
        assert item.summary_row()["requests_to_recover"] == item.recovery_requests
    stationary, discounted = by_name["stationary"], by_name["discounted"]
    # Forgetting buys both a shallower dip and a faster return to the target.
    assert discounted.post_shift_trough > stationary.post_shift_trough
    assert discounted.recovery_requests < stationary.recovery_requests  # type: ignore[operator]
    # Both eventually reach the new optimum: the discount changes the speed, not the destination.
    assert stationary.final_ratio > 0.98
    assert discounted.final_ratio > 0.98


def test_a_policy_the_shift_never_reaches_reports_zero_recovery(
    shifting_environment: SimulatedEnvironment,
) -> None:
    """A mild shift that never pushes the ratio below the target is honestly reported as 0."""
    recovery = run_shift_experiment(
        shifting_environment,
        policies={"linucb": (1.0, learner(shifting_environment))},
        n=6_000,
        seed=25,
        shift_step=1_000,
        window=800,
    )[0]
    assert recovery.post_shift_trough > 0.9
    assert recovery.recovery_requests == 0
    # "0" here means the shift never pushed it below the target, which the table says in words.
    assert recovery.dipped_below_target is False
    assert recovery.summary_row()["requests_to_recover"] == "never fell below target"


def test_the_shift_experiment_needs_the_shift_inside_the_run(
    shifting_environment: SimulatedEnvironment,
) -> None:
    with pytest.raises(ValueError, match="inside the run"):
        run_shift_experiment(
            shifting_environment,
            policies={"a": (1.0, learner(shifting_environment))},
            n=100,
            seed=0,
            shift_step=500,
        )


# --- screening trials -----------------------------------------------------------------------------


def test_screening_rejects_candidates_that_are_genuinely_worse(
    environment: SimulatedEnvironment,
) -> None:
    trials = run_screening_trials(
        environment,
        logger_factory=lambda: learner(environment, "epsilon_greedy", epsilon=0.2),
        candidates={
            "always_haiku": lambda env, _seed: static(env, "haiku"),
            "always_opus": lambda env, _seed: static(env, "opus"),
            "keyword_rule": lambda env, _seed: KeywordRule(
                arms=env.arms,
                dim=env.feature_dim,
                code_arm="opus",
                long_arm="sonnet",
                default_arm="haiku",
                propensity_floor=FLOOR,
                feature_version=env.featurizer.version,
            ),
        },
        n_logs=4_000,
        seeds=(1, 2),
        warmup=800,
        config=ScreeningConfig(bootstrap=120, min_ess=200.0, max_exploration_usd=10_000.0),
    )
    assert trials.truly_worse + trials.truly_better == 6
    assert trials.truly_worse > 0
    assert trials.rejection_rate >= 0.8
    assert 0.0 <= trials.admission_rate <= 1.0
    assert len(trials.reports) == 6
    row = trials.summary_row()
    assert row["truly_worse_candidates"] == trials.truly_worse


# --- the poisoning ablation -----------------------------------------------------------------------


def test_the_poisoning_defenses_are_worth_measuring(
    environment: SimulatedEnvironment,
) -> None:
    stream = environment.workload.sample_indices(4_000, np.random.default_rng(22))
    clean = run_simulation(environment, learner(environment), n=4_000, seed=23, stream=stream)
    results = {
        mode: run_simulation(
            environment,
            learner(environment),
            n=4_000,
            seed=23,
            stream=stream,
            channel=PoisonedFeedbackChannel(attack_rate=0.2, mode=mode),
        )
        for mode in ("none", "clip_only", "robust")
    }
    values = {mode: float(result.log.decision_values.mean()) for mode, result in results.items()}
    baseline = float(clean.log.decision_values.mean())
    # Undefended, hostile feedback costs real policy value; the full pipeline nearly erases it.
    assert values["none"] < values["robust"]
    assert baseline - values["robust"] < baseline - values["none"]
    assert values["robust"] > baseline * 0.95


def test_the_poisoned_channel_clips_and_weights_what_it_reports() -> None:
    rng = np.random.default_rng(24)
    robust = PoisonedFeedbackChannel(attack_rate=1.0, mode="robust")
    reward, weight = robust.observe(0, 0.9, rng)
    assert 0.0 <= reward <= 1.0
    assert 0.0 < weight <= 1.0
    undefended = PoisonedFeedbackChannel(attack_rate=1.0, mode="none")
    raw, raw_weight = undefended.observe(0, 0.9, rng)
    assert raw < -1.0  # the unclipped attacker value reaches the learner
    assert raw_weight == 1.0


def test_the_poisoned_channel_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="attack_rate"):
        PoisonedFeedbackChannel(attack_rate=1.5)
    with pytest.raises(ValueError, match="at least one honest source"):
        PoisonedFeedbackChannel(attack_rate=0.1, honest_sources=0)
