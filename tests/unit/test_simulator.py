"""The offline simulator: the ground truth every claim in this repo is measured against.

If the simulator were not deterministic, none of the reported numbers would be reproducible; if its
quality curves did not actually cross, routing would be trivial and a "30% cheaper at equal quality"
result would mean nothing. Both are checked here:

* one integer seed reproduces the entire workload, and its digest changes when anything about the
  configuration changes;
* bigger arms have flatter difficulty slopes, so each arm is genuinely optimal on part of the
  traffic once cost is priced in;
* the environment's regimes are exact, which is what makes cumulative regret a measurement of the
  policy's decisions rather than of sampling noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from bandit_llm_router.arms import ArmCatalog
from bandit_llm_router.features import Featurizer
from bandit_llm_router.reward import RewardConfig, shape_rewards
from bandit_llm_router.simulator.environment import SimulatedEnvironment, _quadrature_nodes
from bandit_llm_router.simulator.quality import (
    DEFAULT_ARM_QUALITY,
    ArmQualitySpec,
    QualityModel,
    QualityShift,
)
from bandit_llm_router.simulator.workload import FAMILIES, FAMILY_SPECS, Workload

pytestmark = pytest.mark.unit


# --- the workload --------------------------------------------------------------------------------


def test_one_seed_reproduces_the_whole_workload() -> None:
    first = Workload(seed=42, pool_size=128)
    second = Workload(seed=42, pool_size=128)
    assert first.digest == second.digest
    assert [r.context.prompt for r in first.pool] == [r.context.prompt for r in second.pool]
    assert [r.difficulty for r in first.pool] == [r.difficulty for r in second.pool]


def test_the_digest_changes_with_anything_that_changes_the_traffic() -> None:
    base = Workload(seed=42, pool_size=128).digest
    assert Workload(seed=43, pool_size=128).digest != base
    assert Workload(seed=42, pool_size=256).digest != base
    assert Workload(seed=42, pool_size=128, tenants=("solo",)).digest != base
    assert Workload(seed=42, pool_size=128, specs=FAMILY_SPECS[:3]).digest != base


def test_the_pool_covers_the_configured_families() -> None:
    workload = Workload(seed=1, pool_size=1_024)
    families = {request.family for request in workload.pool}
    assert families == set(FAMILIES)
    assert workload.families == FAMILIES
    indices = workload.family_indices()
    assert indices.shape == (1_024,)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < len(FAMILIES)


def test_difficulty_is_latent_but_shapes_the_prompt() -> None:
    """The router may only see difficulty's traces, which is what makes routing contextual."""
    workload = Workload(seed=2, pool_size=512)
    by_family: dict[str, list[tuple[float, int]]] = {}
    for request in workload.pool:
        by_family.setdefault(request.family, []).append(
            (request.difficulty, len(request.context.prompt))
        )
    correlations = []
    for pairs in by_family.values():
        if len(pairs) < 20:
            continue
        difficulty = np.array([item[0] for item in pairs])
        length = np.array([float(item[1]) for item in pairs])
        correlations.append(float(np.corrcoef(difficulty, length)[0, 1]))
    assert min(correlations) > 0.5


def test_every_difficulty_is_a_probability(workload: Workload) -> None:
    difficulties = workload.difficulties()
    assert float(difficulties.min()) >= 0.0
    assert float(difficulties.max()) <= 1.0


def test_code_prompts_carry_a_fence_and_chat_prompts_do_not() -> None:
    workload = Workload(seed=3, pool_size=512)
    for request in workload.pool:
        if request.family == "code":
            assert "```" in request.context.prompt
        if request.family == "small_talk":
            assert "```" not in request.context.prompt


def test_the_request_stream_is_seeded_and_in_range(workload: Workload) -> None:
    indices = workload.sample_indices(500, np.random.default_rng(7))
    again = workload.sample_indices(500, np.random.default_rng(7))
    assert np.array_equal(indices, again)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < len(workload.pool)
    assert workload.sample_indices(0, np.random.default_rng(7)).size == 0
    with pytest.raises(ValueError, match="non-negative"):
        workload.sample_indices(-1, np.random.default_rng(7))


def test_the_workload_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        Workload(seed=1, pool_size=2)
    with pytest.raises(ValueError, match="at least one tenant"):
        Workload(seed=1, pool_size=64, tenants=())


def test_family_weights_are_normalized_not_assumed() -> None:
    weights = sum(spec.weight for spec in FAMILY_SPECS)
    assert weights == pytest.approx(1.0, abs=0.01)


# --- the quality model ---------------------------------------------------------------------------


def test_bigger_arms_degrade_more_slowly_with_difficulty() -> None:
    """This is the premise of contextual routing: the arms' quality curves have to cross."""
    slopes = {spec.arm: spec.difficulty_slope for spec in DEFAULT_ARM_QUALITY}
    assert slopes["haiku"] > slopes["sonnet"] > slopes["opus"]


def test_the_arms_are_near_tied_on_easy_work_and_diverge_on_hard_work() -> None:
    model = QualityModel()
    easy = [model.expected("small_talk", arm, 0.05) for arm in model.arms]
    hard = [model.expected("reasoning", arm, 0.95) for arm in model.arms]
    assert max(easy) - min(easy) < 0.05
    assert max(hard) - min(hard) > 0.3
    assert hard[2] > hard[0]  # opus beats haiku where it matters


def test_expected_quality_is_a_probability_and_falls_with_difficulty() -> None:
    model = QualityModel()
    for family in FAMILIES:
        for arm in model.arms:
            values = [model.expected(family, arm, d / 10) for d in range(11)]
            assert all(0.0 < value < 1.0 for value in values)
            assert values == sorted(values, reverse=True)


def test_the_matrix_form_matches_the_scalar_one() -> None:
    model = QualityModel()
    families = np.array([0, 2, 4], dtype=np.int64)
    difficulties = np.array([0.1, 0.5, 0.9])
    matrix = model.expected_matrix(families, difficulties)
    assert matrix.shape == (3, len(model.arms))
    for row, (family_index, difficulty) in enumerate(zip(families, difficulties, strict=True)):
        for column, arm in enumerate(model.arms):
            assert matrix[row, column] == pytest.approx(
                model.expected(FAMILIES[int(family_index)], arm, float(difficulty))
            )


def test_the_quality_model_validates_its_inputs() -> None:
    model = QualityModel()
    with pytest.raises(ValueError, match="difficulty must be in"):
        model.expected("code", "opus", 1.5)
    with pytest.raises(ValueError, match="same shape"):
        model.expected_matrix(np.array([0, 1], dtype=np.int64), np.array([0.5]))
    with pytest.raises(ValueError, match="family index out of range"):
        model.expected_matrix(np.array([99], dtype=np.int64), np.array([0.5]))
    with pytest.raises(ValueError, match="difficulty must be in"):
        model.expected_matrix(np.array([0], dtype=np.int64), np.array([2.0]))
    with pytest.raises(ValueError, match="at least one arm"):
        QualityModel([])
    with pytest.raises(ValueError, match="missing intercepts"):
        QualityModel(
            [
                ArmQualitySpec(arm="a", intercepts={"code": 1.0}, difficulty_slope=1.0),
                ArmQualitySpec(arm="b", intercepts={"code": 1.0}, difficulty_slope=1.0),
            ]
        )


def test_observed_quality_is_noisy_but_bounded() -> None:
    model = QualityModel()
    expected = np.full(len(model.arms), 0.5)
    samples = np.vstack(
        [model.sample(expected, np.random.default_rng(seed)) for seed in range(500)]
    )
    assert float(samples.min()) >= 0.0
    assert float(samples.max()) <= 1.0
    assert samples.std() > 0.01
    assert samples.mean() == pytest.approx(0.5, abs=0.02)
    batch = model.sample(np.full((4, len(model.arms)), 0.5), np.random.default_rng(1))
    assert batch.shape == (4, len(model.arms))


def test_a_shift_moves_exactly_the_named_arm_and_family() -> None:
    model = QualityModel()
    shifted = model.shifted(
        [QualityShift(at_step=10, family="code", arm="opus", intercept_delta=-2.5)]
    )
    assert shifted.expected("code", "opus", 0.5) < model.expected("code", "opus", 0.5)
    assert shifted.expected("code", "haiku", 0.5) == model.expected("code", "haiku", 0.5)
    assert shifted.expected("reasoning", "opus", 0.5) == model.expected("reasoning", "opus", 0.5)
    with pytest.raises(ValueError, match="unknown family"):
        model.shifted([QualityShift(at_step=1, family="poetry", arm="opus", intercept_delta=-1.0)])


# --- the environment -----------------------------------------------------------------------------


def test_the_environment_precomputes_one_regime_per_shift(
    shifting_environment: SimulatedEnvironment,
) -> None:
    regimes = shifting_environment.regimes
    assert [regime.start_step for regime in regimes] == [0, 1_000]
    assert shifting_environment.regime_at(0) is regimes[0]
    assert shifting_environment.regime_at(999) is regimes[0]
    assert shifting_environment.regime_at(1_000) is regimes[1]
    assert shifting_environment.regime_at(50_000) is regimes[1]
    with pytest.raises(ValueError, match="non-negative"):
        shifting_environment.regime_at(-1)


def test_a_shift_lowers_the_expected_reward_of_the_degraded_arm(
    environment: SimulatedEnvironment, shifting_environment: SimulatedEnvironment
) -> None:
    before = shifting_environment.regimes[0].expected_reward
    after = shifting_environment.regimes[1].expected_reward
    assert before == pytest.approx(environment.regimes[0].expected_reward)
    opus = environment.arms.index("opus")
    families = np.asarray([request.family for request in shifting_environment.workload.pool])
    code_rows = families == "code"
    assert np.all(after[code_rows, opus] <= before[code_rows, opus] + 1e-12)
    assert float(after[code_rows, opus].mean()) < float(before[code_rows, opus].mean())
    other_rows = ~code_rows
    assert after[other_rows, opus] == pytest.approx(before[other_rows, opus])


def test_the_oracle_is_exact_so_regret_measures_decisions(
    environment: SimulatedEnvironment,
) -> None:
    regime = environment.regimes[0]
    assert regime.best_arm.shape == (environment.pool_size,)
    recomputed = np.argmax(regime.expected_reward, axis=1)
    assert np.array_equal(regime.best_arm, recomputed)
    assert regime.best_reward == pytest.approx(regime.expected_reward.max(axis=1))


def test_each_arm_is_optimal_on_part_of_the_traffic(environment: SimulatedEnvironment) -> None:
    """If one arm dominated everywhere, a learned router could not beat a static rule."""
    shares = np.bincount(environment.regimes[0].best_arm, minlength=environment.n_arms)
    assert np.all(shares > 0)
    assert float(shares.max() / shares.sum()) < 0.95


def test_the_environment_exposes_matching_feature_and_cost_matrices(
    environment: SimulatedEnvironment,
) -> None:
    assert environment.feature_matrix.shape == (environment.pool_size, environment.feature_dim)
    assert environment.cost_matrix.shape == (environment.pool_size, environment.n_arms)
    assert environment.ceiling_matrix.shape == environment.cost_matrix.shape
    assert np.all(environment.ceiling_matrix > 0.0)
    assert environment.input_tokens.shape == (environment.pool_size,)
    assert environment.noise_sd.shape == (environment.n_arms,)
    assert environment.expected_rewards(0, 0).shape == (environment.n_arms,)
    assert 0 <= environment.best_arm(0, 0) < environment.n_arms


def test_the_environment_digest_pins_every_input(
    workload: Workload, catalog: ArmCatalog, featurizer: Featurizer
) -> None:
    def build(**overrides: object) -> SimulatedEnvironment:
        values: dict[str, object] = {
            "workload": workload,
            "catalog": catalog,
            "featurizer": featurizer,
        }
        values.update(overrides)
        return SimulatedEnvironment(**values)  # type: ignore[arg-type]

    base = build().digest
    assert build().digest == base
    assert build(reward_config=RewardConfig(cost_weight=0.9)).digest != base
    assert (
        build(
            shifts=(QualityShift(at_step=5, family="code", arm="opus", intercept_delta=-1.0),)
        ).digest
        != base
    )
    assert build(workload=Workload(seed=99, pool_size=128)).digest != base
    payload = build().digest_payload()
    assert payload["workload"] == workload.digest
    assert payload["featurizer"] == featurizer.version


def test_the_environment_refuses_a_quality_model_for_other_arms(
    workload: Workload, catalog: ArmCatalog
) -> None:
    other = QualityModel(
        [
            ArmQualitySpec(
                arm="alpha",
                intercepts=dict.fromkeys(FAMILIES, 1.0),
                difficulty_slope=1.0,
            ),
            ArmQualitySpec(
                arm="beta",
                intercepts=dict.fromkeys(FAMILIES, 1.0),
                difficulty_slope=1.0,
            ),
        ]
    )
    with pytest.raises(ValueError, match="do not match catalog"):
        SimulatedEnvironment(workload=workload, catalog=catalog, quality=other)


def test_a_shift_before_step_zero_is_refused(workload: Workload, catalog: ArmCatalog) -> None:
    with pytest.raises(ValueError, match="cannot start before step 0"):
        SimulatedEnvironment(
            workload=workload,
            catalog=catalog,
            shifts=(QualityShift(at_step=-1, family="code", arm="opus", intercept_delta=-1.0),),
        )


def test_one_outcome_is_the_shaped_reward_of_a_noisy_quality(
    environment: SimulatedEnvironment,
) -> None:
    outcome = environment.outcome(3, 1, 0, standard_normal=0.0)
    regime = environment.regimes[0]
    assert outcome.arm_index == 1
    assert outcome.quality == pytest.approx(regime.expected_quality[3, 1])
    assert outcome.cost_micros == pytest.approx(environment.cost_matrix[3, 1])
    # A zero-noise draw earns the reward *of* the expected quality, which is deliberately not the
    # same as the *expected* reward: the clips censor the noise, so the mean sits slightly lower.
    naive = shape_rewards(
        regime.expected_quality[3:4, 1:2],
        environment.cost_matrix[3:4, 1:2],
        environment.reward_config,
    )
    assert outcome.reward == pytest.approx(float(naive[0, 0]), abs=1e-9)
    assert outcome.reward > regime.expected_reward[3, 1]
    # A positive shock raises quality, a negative one lowers it, and both stay in range.
    high = environment.outcome(3, 1, 0, standard_normal=4.0)
    low = environment.outcome(3, 1, 0, standard_normal=-4.0)
    assert high.quality > outcome.quality
    assert low.quality < outcome.quality
    assert 0.0 <= low.quality <= high.quality <= 1.0


def test_noise_is_pre_drawn_so_the_hot_loop_does_no_rng_work(
    environment: SimulatedEnvironment,
) -> None:
    noise = environment.draw_noise(100, np.random.default_rng(4))
    assert noise.shape == (100,)
    assert np.array_equal(noise, environment.draw_noise(100, np.random.default_rng(4)))


def test_the_expected_reward_is_the_true_mean_of_a_realized_call(
    environment: SimulatedEnvironment,
) -> None:
    """Shaping the reward of the *expected* quality is not the expected reward.

    Both the quality noise and the cost penalty are followed by a clip, so the reward is a censored
    variable. Ignoring that overstates the truth wherever the quality curve runs near 1, which on
    this workload is most of the traffic -- and it would surface as a systematic bias in every
    off-policy estimator, including IPS, whose defining property is having none. The environment
    integrates the noise instead; this test checks the integral against sampling the process.
    """
    regime = environment.regimes[0]
    quality = regime.expected_quality
    penalty = environment.reward_config.cost_weight * np.minimum(
        1.0, environment.cost_matrix / float(environment.reward_config.cost_reference_micros)
    )
    rng = np.random.default_rng(0)
    draws = 2_000
    total = np.zeros_like(quality)
    for _ in range(draws):
        noisy = np.clip(
            quality + rng.normal(size=quality.shape) * environment.noise_sd[np.newaxis, :], 0.0, 1.0
        )
        total += np.clip(noisy - penalty, 0.0, 1.0)
    sampled = total / draws
    assert float(sampled.mean()) == pytest.approx(float(regime.expected_reward.mean()), abs=1e-3)
    assert np.abs(sampled - regime.expected_reward).max() < 0.02

    # And the naive version really is different, so the integral is not decoration.
    naive = shape_rewards(quality, environment.cost_matrix, environment.reward_config)
    assert float(naive.mean()) > float(regime.expected_reward.mean()) + 1e-3


def test_the_noise_quadrature_is_deterministic_and_centred() -> None:
    nodes = _quadrature_nodes()
    assert nodes.shape == (256,)
    assert nodes is _quadrature_nodes()  # cached, so every regime integrates identically
    assert float(nodes.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(nodes.std()) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError, match="read-only"):
        nodes[0] = 0.0


def test_the_in_class_reference_is_the_best_policy_the_model_can_represent(
    environment: SimulatedEnvironment,
) -> None:
    """Regret has to be measured against a reachable target, or it grows linearly by construction.

    A linear bandit over hashed features cannot represent a sigmoid quality curve, so there is an
    irreducible approximation gap. Measuring regret against the unconstrained oracle charges the
    learner for that gap forever: it converges, the gap keeps accruing at a constant rate, and the
    fitted log-log slope creeps toward 1, which reads as "never learned". The in-class reference is
    the greedy policy of the best least-squares linear fit, valued at the *true* expected rewards.
    """
    regime = environment.regimes[0]
    assert regime.linear_best_arm.shape == (environment.pool_size,)
    # It is a real policy, so it can only be as good as the oracle, never better.
    assert np.all(regime.linear_best_reward <= regime.best_reward + 1e-12)
    # And it is strictly worse somewhere: if it were not, the model class would be exact and this
    # reference would be pointless.
    assert float(regime.linear_best_reward.sum()) < float(regime.best_reward.sum())
    # It beats every static arm, so it is a reference worth aiming at rather than a low bar.
    for arm in range(environment.n_arms):
        assert float(regime.linear_best_reward.mean()) > float(
            regime.expected_reward[:, arm].mean()
        )


def test_the_in_class_reference_follows_a_quality_shift(
    severe_shift_environment: SimulatedEnvironment,
) -> None:
    before, after = severe_shift_environment.regimes
    assert not np.array_equal(before.linear_best_arm, after.linear_best_arm)
    assert float(after.linear_best_reward.mean()) < float(before.linear_best_reward.mean())
