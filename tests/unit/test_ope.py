"""Off-policy evaluation: the estimators, and the ways each of them is allowed to be wrong.

These tests use a small synthetic bandit whose exact policy value is computable in closed form, so
"unbiased" and "consistent" are checked against ground truth rather than asserted:

* IPS is unbiased, so its mean over many independent logs converges to the true value.
* SNIPS is biased but bounded: it can never report a value outside the observed reward range.
* DR is consistent when *either* the propensities or the reward model are right, which is the whole
  point of using it -- so both halves of that claim are tested, including with a deliberately
  useless reward model and with deliberately wrong propensities.
* Cross-fitting is what keeps DR from flattering its own reward model, so the out-of-fold property
  is tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from bandit_llm_router.ope.estimators import (
    CrossFitRidgeRewardModel,
    Estimate,
    LoggedBatch,
    doubly_robust,
    effective_sample_size,
    evaluate_policy,
    importance_weights,
    ips,
    snips,
    stack_probabilities,
)

pytestmark = pytest.mark.unit

N_ARMS = 3
DIM = 4


_THETAS = np.array(
    [
        [0.50, 0.20, -0.05, 0.00],
        [0.52, 0.00, 0.25, 0.05],
        [0.48, 0.05, 0.00, 0.30],
    ]
)


def synthetic_log(
    seed: int, *, n: int = 800, floor: float = 0.1, noise: float = 0.05
) -> tuple[LoggedBatch, np.ndarray, np.ndarray]:
    """A log whose expected reward really is linear in the features.

    The first feature is a bias term and the coefficients are chosen so every expected reward lands
    well inside [0, 1] and the Gaussian noise never has to be clipped. Without that,
    ``E[r | x, a]`` would not equal the truth matrix and "IPS is unbiased" could not be tested at
    all -- clipping alone would make it look biased.

    The logging policy is context dependent (it favours the arm a fixed rule prefers, which is *not*
    the best arm) and keeps ``floor`` on every arm, so logged propensities vary across rows the way
    a real router's do. Returns the batch, the truth matrix and the logging distribution per row.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(n, DIM - 1)) * 0.6
    features = np.column_stack([np.ones(n), raw])
    truth = features @ _THETAS.T
    preferred = np.argmax(raw, axis=1)
    logging_probabilities = np.full((n, N_ARMS), floor)
    logging_probabilities[np.arange(n), preferred] += 1.0 - floor * N_ARMS
    cumulative = np.cumsum(logging_probabilities, axis=1)
    actions = (rng.random((n, 1)) > cumulative).sum(axis=1).clip(0, N_ARMS - 1)
    propensities = logging_probabilities[np.arange(n), actions]
    rewards = truth[np.arange(n), actions] + rng.normal(0.0, noise, size=n)
    batch = LoggedBatch(
        features=features,
        actions=actions.astype(np.int64),
        propensities=propensities,
        rewards=rewards,
        n_arms=N_ARMS,
    )
    return batch, truth, logging_probabilities


def target_matrix(rows: int, weights: tuple[float, ...]) -> np.ndarray:
    return np.tile(np.asarray(weights, dtype=np.float64), (rows, 1))


def exact_value(truth: np.ndarray, target: np.ndarray) -> float:
    return float(np.einsum("ij,ij->i", target, truth).mean())


# --- batch validation ---------------------------------------------------------------------------


def test_a_batch_refuses_rows_that_cannot_be_reweighted() -> None:
    features = np.zeros((3, DIM))
    good = {
        "features": features,
        "actions": np.array([0, 1, 2], dtype=np.int64),
        "propensities": np.array([0.4, 0.3, 0.3]),
        "rewards": np.array([0.1, 0.2, 0.3]),
        "n_arms": N_ARMS,
    }
    LoggedBatch(**good)  # type: ignore[arg-type]
    for override, match in (
        ({"propensities": np.array([0.4, 0.0, 0.3])}, "propensities must be positive"),
        ({"propensities": np.array([0.4, 1.5, 0.3])}, "cannot exceed 1"),
        ({"actions": np.array([0, 1, 9], dtype=np.int64)}, "outside the arm set"),
        ({"rewards": np.array([0.1, np.nan, 0.3])}, "rewards contain NaN"),
        ({"features": np.full((3, DIM), np.inf)}, "features contain NaN"),
        ({"n_arms": 1}, "n_arms must be"),
        ({"actions": np.array([0, 1], dtype=np.int64)}, "actions must have shape"),
        ({"features": np.zeros(3)}, "2-D"),
    ):
        with pytest.raises(ValueError, match=match):
            LoggedBatch(**{**good, **override})  # type: ignore[arg-type]


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        LoggedBatch(
            features=np.zeros((0, DIM)),
            actions=np.zeros(0, dtype=np.int64),
            propensities=np.zeros(0),
            rewards=np.zeros(0),
            n_arms=N_ARMS,
        )


def test_the_batch_digest_binds_an_estimate_to_its_data() -> None:
    batch, _, _ = synthetic_log(1, n=50)
    same, _, _ = synthetic_log(1, n=50)
    assert batch.digest == same.digest
    nudged = LoggedBatch(
        features=batch.features,
        actions=batch.actions,
        propensities=batch.propensities,
        rewards=batch.rewards + 1e-9,
        n_arms=batch.n_arms,
    )
    assert nudged.digest != batch.digest
    assert batch.n == 50
    assert batch.dim == DIM


def test_target_probabilities_are_validated() -> None:
    batch, _, _ = synthetic_log(2, n=20)
    with pytest.raises(ValueError, match="must have shape"):
        ips(batch, np.ones((19, N_ARMS)) / N_ARMS)
    with pytest.raises(ValueError, match="sum to 1"):
        ips(batch, np.ones((20, N_ARMS)))
    with pytest.raises(ValueError, match="cannot be negative"):
        ips(batch, target_matrix(20, (-0.5, 0.75, 0.75)))


def test_stack_probabilities_builds_the_matrix_the_estimators_want() -> None:
    rows = [np.array([0.5, 0.25, 0.25]), np.array([0.2, 0.4, 0.4])]
    assert stack_probabilities(rows).shape == (2, 3)
    with pytest.raises(ValueError, match="at least one row"):
        stack_probabilities([])


# --- importance weights and ESS -----------------------------------------------------------------


@given(
    hnp.arrays(np.float64, (32,), elements=st.floats(0.0, 50.0, allow_nan=False, width=64)).filter(
        lambda array: array.sum() > 0.0
    )
)
def test_effective_sample_size_is_bounded_by_the_row_count(weights: np.ndarray) -> None:
    ess = effective_sample_size(weights)
    assert 0.0 <= ess <= weights.size + 1e-9


def test_effective_sample_size_is_exact_for_uniform_weights() -> None:
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert effective_sample_size(np.zeros(10)) == 0.0
    # One dominant weight collapses ESS toward a single row.
    spiky = np.concatenate([[1_000.0], np.ones(99)])
    assert effective_sample_size(spiky) < 2.0


def test_clipping_trades_variance_for_bias_and_reports_how_much() -> None:
    batch, _, _ = synthetic_log(3, n=400, floor=0.1)
    target = target_matrix(batch.n, (1 / 3, 1 / 3, 1 / 3))
    unclipped, fraction = importance_weights(batch, target)
    assert fraction == 0.0
    clipped, clipped_fraction = importance_weights(batch, target, clip=2.0)
    assert float(clipped.max()) <= 2.0
    assert 0.0 < clipped_fraction < 1.0
    assert effective_sample_size(clipped) > effective_sample_size(unclipped)
    with pytest.raises(ValueError, match="clip must be positive"):
        importance_weights(batch, target, clip=0.0)


def test_a_candidate_matching_the_logging_policy_has_unit_weights() -> None:
    batch, _, logging_probabilities = synthetic_log(4, n=200)
    weights, _ = importance_weights(batch, logging_probabilities)
    assert weights == pytest.approx(np.ones(batch.n))
    assert effective_sample_size(weights) == pytest.approx(float(batch.n))


# --- IPS ----------------------------------------------------------------------------------------


def test_ips_is_unbiased_over_independent_logs() -> None:
    """The defining property: average IPS over many logs converges to the exact policy value."""
    target = (0.2, 0.3, 0.5)
    estimates: list[float] = []
    truth_values: list[float] = []
    for seed in range(40):
        batch, truth, _ = synthetic_log(100 + seed, n=400)
        matrix = target_matrix(batch.n, target)
        estimates.append(ips(batch, matrix, bootstrap=0).value)
        truth_values.append(exact_value(truth, matrix))
    assert float(np.mean(estimates)) == pytest.approx(float(np.mean(truth_values)), abs=0.01)


def test_ips_on_the_logging_policy_is_the_mean_logged_reward() -> None:
    batch, _, logging_probabilities = synthetic_log(5, n=300)
    estimate = ips(batch, logging_probabilities, bootstrap=0)
    assert estimate.value == pytest.approx(float(batch.rewards.mean()))
    assert estimate.method == "ips"
    assert estimate.n == batch.n


def test_ips_variance_explodes_when_the_candidate_disagrees_with_the_log() -> None:
    batch, _, logging_probabilities = synthetic_log(6, n=600, floor=0.02)
    agreeing = ips(batch, logging_probabilities, bootstrap=0)
    disagreeing = ips(batch, target_matrix(batch.n, (1 / 3, 1 / 3, 1 / 3)), bootstrap=0)
    assert disagreeing.std_error > agreeing.std_error * 3
    assert disagreeing.ess < agreeing.ess / 3


# --- SNIPS --------------------------------------------------------------------------------------


def test_snips_can_never_leave_the_reward_range() -> None:
    """IPS can; that difference is the reason both are reported."""
    batch, _, _ = synthetic_log(7, n=300, floor=0.02)
    target = target_matrix(batch.n, (1.0, 0.0, 0.0))
    self_normalized = snips(batch, target, bootstrap=0)
    assert float(batch.rewards.min()) <= self_normalized.value <= float(batch.rewards.max())
    assert self_normalized.method == "snips"


def test_snips_is_close_to_the_truth_where_ips_is_noisy() -> None:
    batch, truth, _ = synthetic_log(8, n=800, floor=0.05)
    target = target_matrix(batch.n, (0.45, 0.45, 0.1))
    exact = exact_value(truth, target)
    assert abs(snips(batch, target, bootstrap=0).value - exact) < 0.2


def test_snips_agrees_with_ips_when_the_weights_are_all_one() -> None:
    batch, _, target = synthetic_log(9, n=200)
    assert snips(batch, target, bootstrap=0).value == pytest.approx(
        ips(batch, target, bootstrap=0).value
    )


# --- the cross-fitted reward model --------------------------------------------------------------


def test_cross_fitting_predicts_each_row_from_models_that_never_saw_it() -> None:
    batch, _, _ = synthetic_log(10, n=200)
    model = CrossFitRidgeRewardModel(ridge=1.0, folds=4)
    predictions = model.fit_predict(batch, np.random.default_rng(0))
    assert predictions.shape == (batch.n, N_ARMS)
    # An in-sample fit would follow the rewards more tightly than a genuinely held-out one.
    in_sample = np.zeros_like(predictions)
    for arm in range(N_ARMS):
        selector = batch.actions == arm
        gram = batch.features[selector].T @ batch.features[selector] + np.eye(DIM)
        theta = np.linalg.solve(gram, batch.features[selector].T @ batch.rewards[selector])
        in_sample[:, arm] = batch.features @ theta
    chosen = np.arange(batch.n), batch.actions
    assert (
        np.abs(batch.rewards - predictions[chosen]).mean()
        > np.abs(batch.rewards - in_sample[chosen]).mean()
    )


def test_the_reward_model_recovers_the_truth_it_was_given() -> None:
    batch, truth, _ = synthetic_log(11, n=2_000, noise=0.01, floor=0.3)
    predictions = CrossFitRidgeRewardModel(ridge=1e-3).fit_predict(batch, np.random.default_rng(1))
    assert predictions == pytest.approx(truth, abs=0.06)


def test_the_reward_model_falls_back_when_an_arm_was_never_taken() -> None:
    batch, _, _ = synthetic_log(12, n=120, floor=0.05)
    forced = LoggedBatch(
        features=batch.features,
        actions=np.where(batch.actions == 2, 0, batch.actions),
        propensities=batch.propensities,
        rewards=batch.rewards,
        n_arms=N_ARMS,
    )
    predictions = CrossFitRidgeRewardModel(folds=3).fit_predict(forced, np.random.default_rng(2))
    assert np.all(np.isfinite(predictions))
    # The never-chosen arm falls back to a constant out-of-fold mean per fold.
    assert len(np.unique(np.round(predictions[:, 2], 12))) <= 3


def test_the_reward_model_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="ridge must be positive"):
        CrossFitRidgeRewardModel(ridge=0.0)
    with pytest.raises(ValueError, match="folds must be >= 2"):
        CrossFitRidgeRewardModel(folds=1)


# --- doubly robust ------------------------------------------------------------------------------


def test_dr_is_consistent_when_only_the_propensities_are_right() -> None:
    """A useless reward model must not stop DR from converging: that is the "doubly" part."""
    target = (0.2, 0.3, 0.5)
    errors: list[float] = []
    for seed in range(30):
        batch, truth, _ = synthetic_log(200 + seed, n=500)
        matrix = target_matrix(batch.n, target)
        useless = np.zeros((batch.n, N_ARMS))
        estimate = doubly_robust(batch, matrix, useless, bootstrap=0)
        errors.append(estimate.value - exact_value(truth, matrix))
    assert float(np.mean(errors)) == pytest.approx(0.0, abs=0.01)


def test_dr_is_consistent_when_only_the_reward_model_is_right() -> None:
    """Propensities wrong by a factor of two; a correct q-hat still rescues the estimate."""
    batch, truth, _ = synthetic_log(13, n=1_500)
    wrong = LoggedBatch(
        features=batch.features,
        actions=batch.actions,
        propensities=np.clip(batch.propensities * 0.5, 1e-6, 1.0),
        rewards=batch.rewards,
        n_arms=N_ARMS,
    )
    matrix = target_matrix(batch.n, (0.2, 0.3, 0.5))
    exact = exact_value(truth, matrix)
    biased = ips(wrong, matrix, bootstrap=0).value
    rescued = doubly_robust(wrong, matrix, truth, bootstrap=0).value
    assert abs(biased - exact) > 0.2
    assert abs(rescued - exact) < 0.05


def test_dr_has_lower_variance_than_ips_with_a_decent_reward_model() -> None:
    batch, _, _ = synthetic_log(14, n=800, floor=0.05)
    matrix = target_matrix(batch.n, (0.45, 0.45, 0.1))
    q_hat = CrossFitRidgeRewardModel().fit_predict(batch, np.random.default_rng(3))
    assert (
        doubly_robust(batch, matrix, q_hat, bootstrap=0).std_error
        < ips(batch, matrix, bootstrap=0).std_error
    )


def test_dr_validates_its_reward_model() -> None:
    batch, _, _ = synthetic_log(15, n=40)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    with pytest.raises(ValueError, match="same shape"):
        doubly_robust(batch, matrix, np.zeros((batch.n, N_ARMS + 1)))
    with pytest.raises(ValueError, match="NaN"):
        doubly_robust(batch, matrix, np.full((batch.n, N_ARMS), np.nan))


# --- intervals ----------------------------------------------------------------------------------


def test_bootstrap_intervals_bracket_the_point_estimate() -> None:
    batch, _, _ = synthetic_log(16, n=400)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    for estimate in (
        ips(batch, matrix, bootstrap=300),
        snips(batch, matrix, bootstrap=300),
        doubly_robust(batch, matrix, np.zeros((batch.n, N_ARMS)), bootstrap=300),
    ):
        assert estimate.ci_low <= estimate.value <= estimate.ci_high
        assert estimate.ci_high > estimate.ci_low


def test_without_the_bootstrap_the_interval_is_the_analytic_one() -> None:
    batch, _, _ = synthetic_log(17, n=400)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    estimate = ips(batch, matrix, bootstrap=0)
    half = 1.959963984540054 * estimate.std_error
    assert estimate.ci_low == pytest.approx(estimate.value - half)
    assert estimate.ci_high == pytest.approx(estimate.value + half)


def test_dr_intervals_cover_the_truth_at_roughly_the_nominal_rate() -> None:
    """Coverage is measured, not assumed; bench/bench_ope.py reports it at full size."""
    target = (0.25, 0.35, 0.4)
    covered = 0
    trials = 60
    for seed in range(trials):
        batch, truth, _ = synthetic_log(300 + seed, n=600)
        matrix = target_matrix(batch.n, target)
        exact = exact_value(truth, matrix)
        estimate = doubly_robust(
            batch,
            matrix,
            CrossFitRidgeRewardModel().fit_predict(batch, np.random.default_rng(seed)),
            bootstrap=200,
            rng=np.random.default_rng(seed + 1),
        )
        covered += int(estimate.ci_low <= exact <= estimate.ci_high)
    assert covered / trials >= 0.85


def autocorrelated_log(
    seed: int, *, n: int = 2_000, rho: float = 0.99, amplitude: float = 0.25
) -> LoggedBatch:
    """A log whose per-row estimator terms are serially dependent, not independent.

    The reward *level* follows an AR(1) process in row order, so the sample mean's variance is
    inflated over the i.i.d. value by the sum of the autocovariances -- which is what a bootstrap
    over rows cannot see, because resampling rows independently destroys the ordering that carries
    the dependence. The marginal distribution of a single row is unchanged, which is the point.

    Note what would *not* work as a test: drifting the logging *policy* while the reward stays
    stationary. Importance weighting is conditionally unbiased, so a drifting propensity changes the
    variance of each row's term without changing its mean, leaves the autocovariances near zero, and
    the two bootstraps agree. Dependence in the mean is what the block bootstrap exists for.
    """
    rng = np.random.default_rng(seed)
    features = np.column_stack([np.ones(n), rng.uniform(-1.0, 1.0, size=(n, DIM - 1)) * 0.6])
    actions = rng.integers(0, N_ARMS, size=n)
    innovations = rng.normal(0.0, 1.0, size=n)
    latent = np.empty(n, dtype=np.float64)
    latent[0] = innovations[0]
    for index in range(1, n):
        latent[index] = rho * latent[index - 1] + np.sqrt(1.0 - rho**2) * innovations[index]
    rewards = np.clip(0.5 + amplitude * latent + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    return LoggedBatch(
        features=features,
        actions=actions.astype(np.int64),
        propensities=np.full(n, 1.0 / N_ARMS),
        rewards=rewards,
        n_arms=N_ARMS,
    )


def _width(estimate: Estimate) -> float:
    return estimate.ci_high - estimate.ci_low


def test_the_row_bootstrap_understates_the_interval_on_a_serially_dependent_log() -> None:
    """The defect the block bootstrap exists for, stated as a number.

    A log written by a learner that was still updating is not i.i.d. in its rows. Resampling rows
    independently therefore reports an interval that would be right for a frozen logger and is too
    narrow here. Blocks of contiguous rows keep the dependence inside the resample.
    """
    batch = autocorrelated_log(11)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    q_hat = np.zeros((batch.n, N_ARMS))
    rows = doubly_robust(batch, matrix, q_hat, bootstrap=400, rng=np.random.default_rng(1))
    blocks = doubly_robust(
        batch, matrix, q_hat, bootstrap=400, rng=np.random.default_rng(1), block_size=200
    )
    assert _width(blocks) > 4.0 * _width(rows)
    assert blocks.value == pytest.approx(rows.value)


def test_the_block_bootstrap_agrees_with_the_row_bootstrap_when_rows_are_independent() -> None:
    """It is not a blanket widening: with no serial dependence the two intervals match."""
    batch, _, _ = synthetic_log(12, n=2_000)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    q_hat = np.zeros((batch.n, N_ARMS))
    rows = doubly_robust(batch, matrix, q_hat, bootstrap=400, rng=np.random.default_rng(2))
    blocks = doubly_robust(
        batch, matrix, q_hat, bootstrap=400, rng=np.random.default_rng(2), block_size=200
    )
    assert _width(blocks) == pytest.approx(_width(rows), rel=0.3)


def test_a_block_size_of_one_is_exactly_the_row_bootstrap() -> None:
    batch, _, _ = synthetic_log(13, n=300)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    plain = ips(batch, matrix, bootstrap=100, rng=np.random.default_rng(3))
    unit = ips(batch, matrix, bootstrap=100, rng=np.random.default_rng(3), block_size=1)
    assert (unit.ci_low, unit.ci_high) == (plain.ci_low, plain.ci_high)


def test_every_row_can_be_drawn_by_the_circular_block_bootstrap() -> None:
    """Wrapping matters: without it the last rows of the log would be systematically under-drawn."""
    from bandit_llm_router.ope.estimators import _resample_indices

    rng = np.random.default_rng(4)
    seen: set[int] = set()
    for _ in range(400):
        seen.update(int(value) for value in _resample_indices(50, 7, rng))
    assert seen == set(range(50))
    assert all(_resample_indices(50, 7, rng).shape == (50,) for _ in range(5))


def test_a_block_size_below_one_is_refused() -> None:
    batch, _, _ = synthetic_log(14, n=200)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    for call in (
        lambda: ips(batch, matrix, block_size=0),
        lambda: snips(batch, matrix, block_size=0),
        lambda: doubly_robust(batch, matrix, np.zeros((batch.n, N_ARMS)), block_size=0),
        lambda: evaluate_policy(batch, matrix, block_size=0),
    ):
        with pytest.raises(ValueError, match="block_size"):
            call()


def test_the_report_passes_the_block_size_to_all_three_estimators() -> None:
    batch = autocorrelated_log(15)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    rows = evaluate_policy(batch, matrix, bootstrap=200, rng=np.random.default_rng(5))
    blocks = evaluate_policy(
        batch, matrix, bootstrap=200, rng=np.random.default_rng(5), block_size=200
    )
    for method in ("ips", "snips", "dr"):
        assert _width(getattr(blocks, method)) > _width(getattr(rows, method))


def test_a_single_row_has_no_standard_error_to_report() -> None:
    batch = LoggedBatch(
        features=np.ones((1, DIM)),
        actions=np.zeros(1, dtype=np.int64),
        propensities=np.array([0.5]),
        rewards=np.array([0.6]),
        n_arms=N_ARMS,
    )
    matrix = target_matrix(1, (0.5, 0.25, 0.25))
    assert ips(batch, matrix, bootstrap=0).std_error == 0.0
    assert snips(batch, matrix, bootstrap=0).std_error == 0.0


# --- the combined report ------------------------------------------------------------------------


def test_the_report_runs_all_three_estimators_on_one_reward_model() -> None:
    batch, truth, _ = synthetic_log(18, n=600)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    report = evaluate_policy(batch, matrix, bootstrap=100)
    assert {report.ips.method, report.snips.method, report.dr.method} == {"ips", "snips", "dr"}
    assert report.n == batch.n
    assert report.min_propensity == pytest.approx(float(batch.propensities.min()))
    assert report.mean_logged_reward == pytest.approx(float(batch.rewards.mean()))
    exact = exact_value(truth, matrix)
    assert abs(report.dr.value - exact) < 0.05
    payload = report.as_dict()
    assert payload["n"] == batch.n
    assert isinstance(payload["dr"], dict)


def test_the_report_accepts_a_supplied_reward_model() -> None:
    batch, truth, _ = synthetic_log(19, n=300)
    matrix = target_matrix(batch.n, (0.3, 0.3, 0.4))
    shared = evaluate_policy(batch, matrix, bootstrap=0, q_hat=truth)
    assert shared.dr.value == pytest.approx(doubly_robust(batch, matrix, truth, bootstrap=0).value)


def test_an_estimate_serializes_to_plain_numbers() -> None:
    payload = Estimate(
        method="dr", value=0.5, std_error=0.01, ci_low=0.48, ci_high=0.52, n=10, ess=9.5
    ).as_dict()
    assert payload == {
        "method": "dr",
        "value": 0.5,
        "std_error": 0.01,
        "ci_low": 0.48,
        "ci_high": 0.52,
        "n": 10,
        "ess": 9.5,
        "clipped_fraction": 0.0,
    }
