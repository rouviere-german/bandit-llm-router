"""Policies: the distribution invariants every routing decision depends on.

Three things here are load-bearing for the whole repo and are therefore property-tested rather than
spot-checked:

1. Every policy returns a probability distribution that respects its propensity floor. If it did
   not, a logged propensity could be zero and the log would be unusable for off-policy evaluation.
2. :func:`restrict_distribution` conserves probability mass when arms are blocked. The realized
   propensity is what gets logged, so mass that goes missing is a silently wrong importance weight.
3. The Sherman-Morrison inverse the ridge posteriors maintain incrementally agrees with a fresh
   solve. That is an optimization, and an optimization that drifts would corrupt every confidence
   width.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from bandit_llm_router.ml.serialization import dumps_npz
from bandit_llm_router.policies.base import (
    Decision,
    NoArmAvailableError,
    Policy,
    apply_propensity_floor,
    greedy_probabilities,
    policy_version,
    restrict_distribution,
    sample_action,
    sample_action_from_uniform,
    validate_distribution,
)
from bandit_llm_router.policies.baselines import KeywordRule, StaticArm, StaticPolicy, feature_index
from bandit_llm_router.policies.factory import (
    POLICY_KINDS,
    PolicyHyperparameters,
    build_policy,
    policy_from_registry_files,
    policy_from_snapshot,
)
from bandit_llm_router.policies.linear import DiscountedRidge
from bandit_llm_router.policies.linear_bandit import (
    EpsilonGreedy,
    LinearThompson,
    LinUCB,
)

pytestmark = pytest.mark.unit

ARMS = ("haiku", "sonnet", "opus")
DIM = 6


def unit_vectors(dim: int = DIM) -> st.SearchStrategy[np.ndarray]:
    """Bounded, finite feature rows, like the featurizer's contract guarantees."""
    return hnp.arrays(
        np.float64,
        (dim,),
        elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False, width=64),
    )


def simplex(n: int) -> st.SearchStrategy[np.ndarray]:
    """Distributions over ``n`` arms with no zero total."""
    return (
        hnp.arrays(np.float64, (n,), elements=st.floats(0.0, 1.0, allow_nan=False, width=64))
        .filter(lambda array: array.sum() > 1e-6)
        .map(lambda array: array / array.sum())
    )


# --- distribution algebra ---------------------------------------------------------------------


@given(simplex(4), st.floats(0.0, 0.24))
def test_the_floor_lifts_every_arm_and_keeps_a_distribution(
    probabilities: np.ndarray, floor: float
) -> None:
    floored = apply_propensity_floor(probabilities, floor)
    assert floored.sum() == pytest.approx(1.0)
    assert float(floored.min()) >= floor - 1e-12
    # The mixture is affine and monotone, so the original ordering survives (weakly: two
    # probabilities that differ by less than the mixture's rounding can end up equal).
    order = np.argsort(probabilities, kind="stable")
    assert np.all(np.diff(floored[order]) >= -1e-12)


@given(simplex(3))
def test_a_zero_floor_leaves_the_distribution_alone(probabilities: np.ndarray) -> None:
    assert apply_propensity_floor(probabilities, 0.0) == pytest.approx(probabilities)


def test_an_impossible_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="impossible"):
        apply_propensity_floor(np.array([0.5, 0.5]), 0.6)
    with pytest.raises(ValueError, match="non-negative"):
        apply_propensity_floor(np.array([0.5, 0.5]), -0.1)


@pytest.mark.parametrize(
    ("array", "match"),
    [
        (np.array([]), "non-empty"),
        (np.array([[0.5, 0.5]]), "non-empty"),
        (np.array([np.nan, 1.0]), "NaN"),
        (np.array([-0.2, 1.2]), "negative"),
        (np.array([0.3, 0.3]), "sums to"),
    ],
)
def test_validate_distribution_rejects_non_distributions(array: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_distribution(array)


def test_validate_distribution_enforces_the_floor() -> None:
    with pytest.raises(ValueError, match="propensity floor"):
        validate_distribution(np.array([0.001, 0.999]), floor=0.01)


@given(hnp.arrays(np.float64, (4,), elements=st.floats(-10.0, 10.0, allow_nan=False, width=64)))
def test_greedy_puts_all_mass_on_maxima(values: np.ndarray) -> None:
    probabilities = greedy_probabilities(values)
    assert probabilities.sum() == pytest.approx(1.0)
    best = values == values.max()
    assert np.all(probabilities[~best] == 0.0)
    assert float(probabilities[best].sum()) == pytest.approx(1.0)


def test_greedy_splits_exact_ties_evenly() -> None:
    assert greedy_probabilities(np.array([1.0, 1.0, 0.0])) == pytest.approx([0.5, 0.5, 0.0])


def test_greedy_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN"):
        greedy_probabilities(np.array([1.0, np.inf]))


@given(
    simplex(4),
    hnp.arrays(np.bool_, (4,), elements=st.booleans()),
)
def test_restriction_conserves_mass_and_empties_blocked_arms(
    probabilities: np.ndarray, blocked: np.ndarray
) -> None:
    assume(not blocked.all())
    fallback = (2, 0, 3, 1)
    realized = restrict_distribution(probabilities, blocked, fallback)
    assert realized.sum() == pytest.approx(1.0)
    assert np.all(realized[blocked] == 0.0)
    # Mass only ever moves *to* the substitute, so an unblocked arm never loses probability.
    assert np.all(realized[~blocked] >= probabilities[~blocked] - 1e-12)


def test_restriction_sends_mass_to_the_first_available_fallback() -> None:
    probabilities = np.array([0.5, 0.3, 0.2])
    realized = restrict_distribution(probabilities, np.array([True, False, False]), (0, 1, 2))
    assert realized == pytest.approx([0.0, 0.8, 0.2])
    realized = restrict_distribution(probabilities, np.array([True, True, False]), (0, 1, 2))
    assert realized == pytest.approx([0.0, 0.0, 1.0])


def test_restriction_refuses_when_every_arm_is_blocked() -> None:
    with pytest.raises(NoArmAvailableError):
        restrict_distribution(np.array([0.5, 0.5]), np.array([True, True]), (0, 1))


def test_restriction_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        restrict_distribution(np.array([0.5, 0.5]), np.array([True]), (0, 1))
    with pytest.raises(ValueError, match="permutation"):
        restrict_distribution(np.array([0.5, 0.5]), np.array([True, False]), (0, 0))


@given(simplex(5), st.floats(0.0, 0.999_999))
def test_inverse_cdf_sampling_only_returns_reachable_arms(
    probabilities: np.ndarray, uniform: float
) -> None:
    arm = sample_action_from_uniform(probabilities, uniform)
    assert 0 <= arm < probabilities.size
    assert probabilities[arm] > 0.0


def test_sampling_frequencies_match_the_distribution() -> None:
    probabilities = np.array([0.1, 0.6, 0.3])
    rng = np.random.default_rng(3)
    draws = np.array([sample_action(probabilities, rng) for _ in range(20_000)])
    shares = np.bincount(draws, minlength=3) / draws.size
    assert shares == pytest.approx(probabilities, abs=0.015)


def test_sampling_rejects_a_uniform_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="uniform"):
        sample_action_from_uniform(np.array([0.5, 0.5]), 1.0)


def test_a_decision_refuses_a_zero_propensity() -> None:
    probabilities = np.array([1.0, 0.0])
    Decision(arm_index=0, propensity=1.0, probabilities=probabilities)
    with pytest.raises(ValueError, match="propensity must be positive"):
        Decision(arm_index=1, propensity=0.0, probabilities=probabilities)
    with pytest.raises(ValueError, match="outside the distribution"):
        Decision(arm_index=5, propensity=0.5, probabilities=probabilities)


def test_policy_version_is_stable_and_configuration_sensitive() -> None:
    first = policy_version("linucb", {"alpha": 0.5, "floor": 0.02})
    assert first == policy_version("linucb", {"floor": 0.02, "alpha": 0.5})
    assert first != policy_version("linucb", {"alpha": 0.6, "floor": 0.02})
    assert first.startswith("linucb-")


# --- discounted ridge posteriors ---------------------------------------------------------------


def test_the_maintained_inverse_matches_a_fresh_solve() -> None:
    """Sherman-Morrison is an optimization; a drifting inverse would corrupt every width."""
    posterior = DiscountedRidge(3, DIM, ridge=1.0, refresh_every=10_000)
    rng = np.random.default_rng(5)
    for step in range(400):
        features = rng.normal(size=DIM)
        posterior.update(step % 3, features, float(rng.random()))
    for arm in range(3):
        gram = posterior.arrays()["gram"][arm]
        assert posterior.inverse(arm) == pytest.approx(np.linalg.inv(gram), abs=1e-8)
        assert posterior.theta(arm) == pytest.approx(
            np.linalg.solve(gram, posterior.arrays()["rhs"][arm]), abs=1e-8
        )


def test_a_periodic_refresh_recomputes_the_inverse_exactly() -> None:
    posterior = DiscountedRidge(2, DIM, refresh_every=8)
    rng = np.random.default_rng(6)
    for _ in range(40):
        posterior.update(0, rng.normal(size=DIM), 0.5)
    gram = posterior.arrays()["gram"][0]
    assert posterior.inverse(0) == pytest.approx(np.linalg.inv(gram), abs=1e-10)


def test_ridge_recovers_a_linear_reward() -> None:
    posterior = DiscountedRidge(2, DIM, ridge=1e-3)
    truth = np.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.5])
    rng = np.random.default_rng(7)
    for _ in range(3_000):
        features = rng.normal(size=DIM)
        posterior.update(0, features, float(features @ truth))
    assert posterior.theta(0) == pytest.approx(truth, abs=0.02)


def test_widths_shrink_where_evidence_accumulates() -> None:
    posterior = DiscountedRidge(2, DIM)
    features = np.zeros(DIM)
    features[0] = 1.0
    before = posterior.widths(features)
    for _ in range(50):
        posterior.update(0, features, 1.0)
    after = posterior.widths(features)
    assert after[0] < before[0]
    assert after[1] == pytest.approx(before[1])


def test_the_discount_lets_a_posterior_forget_and_re_explore() -> None:
    """A discounted posterior must track a changed world; a stationary one must not."""
    features = np.zeros(DIM)
    features[0] = 1.0
    discounted = DiscountedRidge(2, DIM, discount=0.99)
    stationary = DiscountedRidge(2, DIM, discount=1.0)
    for posterior in (discounted, stationary):
        for _ in range(500):
            posterior.update(0, features, 1.0)
    width_after_learning = float(discounted.widths(features)[0])
    for posterior in (discounted, stationary):
        for _ in range(500):
            posterior.update(0, features, 0.0)
    # The discounted mean has moved most of the way to the new reward; the stationary one is stuck
    # near the average of the two regimes.
    assert float(discounted.means(features)[0]) < 0.15
    assert float(stationary.means(features)[0]) > 0.4
    # Discounted mass is bounded, so uncertainty stops collapsing: exploration can restart.
    assert float(discounted.mass[0]) < 500.0
    assert float(stationary.mass[0]) == pytest.approx(1_000.0)
    assert float(discounted.widths(features)[0]) >= width_after_learning * 0.5


def test_matrix_predictions_agree_with_the_per_row_ones() -> None:
    posterior = DiscountedRidge(3, DIM)
    rng = np.random.default_rng(8)
    for step in range(60):
        posterior.update(step % 3, rng.normal(size=DIM), float(rng.random()))
    rows = rng.normal(size=(7, DIM))
    assert posterior.means_matrix(rows) == pytest.approx(
        np.vstack([posterior.means(row) for row in rows])
    )
    assert posterior.widths_matrix(rows) == pytest.approx(
        np.vstack([posterior.widths(row) for row in rows])
    )


def test_posterior_round_trips_through_its_arrays() -> None:
    posterior = DiscountedRidge(2, DIM, ridge=0.7, discount=0.995)
    rng = np.random.default_rng(9)
    for _ in range(30):
        posterior.update(1, rng.normal(size=DIM), 0.3)
    restored = DiscountedRidge.from_arrays(posterior.arrays(), ridge=0.7, discount=0.995)
    features = rng.normal(size=DIM)
    assert restored.means(features) == pytest.approx(posterior.means(features))
    assert restored.widths(features) == pytest.approx(posterior.widths(features))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_arms": 1}, "n_arms"),
        ({"dim": 0}, "dim"),
        ({"ridge": 0.0}, "ridge"),
        ({"discount": 1.5}, "discount"),
        ({"refresh_every": 0}, "refresh_every"),
    ],
)
def test_posterior_validates_its_configuration(kwargs: dict[str, float], match: str) -> None:
    defaults: dict[str, float] = {"n_arms": 2, "dim": DIM}
    defaults.update(kwargs)
    n_arms = int(defaults.pop("n_arms"))
    dim = int(defaults.pop("dim"))
    with pytest.raises(ValueError, match=match):
        DiscountedRidge(n_arms, dim, **defaults)  # type: ignore[arg-type]


def test_posterior_validates_an_update() -> None:
    posterior = DiscountedRidge(2, DIM)
    with pytest.raises(ValueError, match="out of range"):
        posterior.update(5, np.zeros(DIM), 0.5)
    with pytest.raises(ValueError, match="shape"):
        posterior.update(0, np.zeros(DIM + 1), 0.5)
    with pytest.raises(ValueError, match="weight"):
        posterior.update(0, np.zeros(DIM), 0.5, weight=-1.0)
    with pytest.raises(ValueError, match="finite"):
        posterior.update(0, np.zeros(DIM), float("nan"))


def test_from_arrays_rejects_mismatched_shapes() -> None:
    arrays = DiscountedRidge(2, DIM).arrays()
    with pytest.raises(ValueError, match="rhs and mass"):
        DiscountedRidge.from_arrays({**arrays, "mass": np.zeros(5)}, ridge=1.0, discount=1.0)
    with pytest.raises(ValueError, match="gram must have shape"):
        DiscountedRidge.from_arrays(
            {"gram": np.zeros((2, 3)), "rhs": np.zeros((2, 3)), "mass": np.zeros(2)},
            ridge=1.0,
            discount=1.0,
        )


# --- the three learning policies ---------------------------------------------------------------


def build(kind: str, **overrides: object) -> LinUCB | LinearThompson | EpsilonGreedy:
    common: dict[str, object] = {
        "arms": ARMS,
        "dim": DIM,
        "propensity_floor": 0.05,
        "feature_version": "ctx-test",
    }
    common.update(overrides)
    if kind == "linucb":
        return LinUCB(**common)  # type: ignore[arg-type]
    if kind == "thompson":
        return LinearThompson(**common)  # type: ignore[arg-type]
    return EpsilonGreedy(**common)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["linucb", "thompson", "epsilon_greedy"])
@given(features=unit_vectors())
def test_every_policy_returns_a_floored_distribution(kind: str, features: np.ndarray) -> None:
    policy = build(kind)
    probabilities = policy.probabilities(features, np.random.default_rng(0))
    assert probabilities.shape == (len(ARMS),)
    assert probabilities.sum() == pytest.approx(1.0)
    assert float(probabilities.min()) >= policy.propensity_floor - 1e-12


@pytest.mark.parametrize("kind", ["linucb", "thompson", "epsilon_greedy"])
def test_learned_policies_satisfy_the_policy_protocol(kind: str) -> None:
    assert isinstance(build(kind), Policy)


@pytest.mark.parametrize("kind", ["linucb", "epsilon_greedy"])
def test_deterministic_policies_agree_between_single_and_batch(kind: str) -> None:
    policy = build(kind)
    rng = np.random.default_rng(10)
    for step in range(30):
        policy.update(rng.normal(size=DIM) * 0.3, step % 3, float(rng.random()))
    rows = rng.normal(size=(9, DIM)) * 0.3
    batch = policy.probabilities_matrix(rows, rng)
    single = np.vstack([policy.probabilities(row, rng) for row in rows])
    assert batch == pytest.approx(single)


def test_thompson_batch_rows_are_floored_distributions() -> None:
    policy = build("thompson", mc_samples=16)
    rows = np.random.default_rng(11).normal(size=(12, DIM)) * 0.3
    matrix = policy.probabilities_matrix(rows, np.random.default_rng(12))
    assert matrix.shape == (12, len(ARMS))
    assert matrix.sum(axis=1) == pytest.approx(np.ones(12))
    assert float(matrix.min()) >= policy.propensity_floor - 1e-12


def test_thompson_probabilities_are_monte_carlo_not_one_hot() -> None:
    """Logging an argmax at probability 1 would make every later OPE estimate wrong."""
    policy = build("thompson", mc_samples=64)
    features = np.ones(DIM) * 0.2
    probabilities = policy.probabilities(features, np.random.default_rng(13))
    assert float(probabilities.max()) < 1.0
    # With no evidence the posteriors are symmetric, so no arm should dominate outright.
    assert float(probabilities.min()) > 0.1


def test_thompson_concentrates_on_the_arm_it_learns_is_best() -> None:
    policy = build("thompson", mc_samples=64, posterior_sd=0.1)
    features = np.zeros(DIM)
    features[0] = 1.0
    for _ in range(300):
        policy.update(features, 2, 1.0)
        policy.update(features, 0, 0.0)
        policy.update(features, 1, 0.0)
    probabilities = policy.probabilities(features, np.random.default_rng(14))
    assert int(np.argmax(probabilities)) == 2
    assert float(probabilities[2]) > 0.8


def test_linucb_prefers_the_arm_with_the_better_estimate() -> None:
    policy = build("linucb", alpha=0.0)
    features = np.zeros(DIM)
    features[1] = 1.0
    for _ in range(200):
        policy.update(features, 1, 0.9)
        policy.update(features, 0, 0.1)
        policy.update(features, 2, 0.1)
    probabilities = policy.probabilities(features, np.random.default_rng(15))
    assert int(np.argmax(probabilities)) == 1


def test_linucb_alpha_buys_optimism_about_unexplored_arms() -> None:
    optimistic = build("linucb", alpha=2.0)
    greedy = build("linucb", alpha=0.0)
    features = np.zeros(DIM)
    features[0] = 1.0
    for policy in (optimistic, greedy):
        for _ in range(50):
            policy.update(features, 0, 0.6)
    # The greedy policy locks onto the only arm it has evidence for; the optimistic one does not.
    assert int(np.argmax(greedy.probabilities(features, np.random.default_rng(0)))) == 0
    scores = optimistic.scores(features)  # type: ignore[union-attr]
    assert float(scores[1]) > float(scores[0])


def test_epsilon_greedy_spends_exactly_epsilon_on_exploration() -> None:
    policy = build("epsilon_greedy", epsilon=0.3, propensity_floor=0.0)
    features = np.zeros(DIM)
    features[0] = 1.0
    for _ in range(100):
        policy.update(features, 2, 1.0)
    probabilities = policy.probabilities(features, np.random.default_rng(16))
    assert probabilities[2] == pytest.approx(1.0 - 0.3 + 0.3 / 3)
    assert probabilities[0] == pytest.approx(0.3 / 3)


@pytest.mark.parametrize(
    ("kind", "kwargs", "match"),
    [
        ("linucb", {"alpha": -1.0}, "alpha"),
        ("thompson", {"posterior_sd": 0.0}, "posterior_sd"),
        ("thompson", {"mc_samples": 0}, "mc_samples"),
        ("epsilon_greedy", {"epsilon": 1.5}, "epsilon"),
        ("linucb", {"arms": ("only",)}, "at least two arms"),
        ("linucb", {"arms": ("a", "a")}, "unique"),
        ("linucb", {"propensity_floor": 0.9}, "propensity_floor"),
    ],
)
def test_policies_validate_their_configuration(
    kind: str, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build(kind, **kwargs)


def test_policies_reject_a_badly_shaped_batch() -> None:
    policy = build("linucb")
    with pytest.raises(ValueError, match="shape"):
        policy.probabilities_matrix(np.zeros((3, DIM + 1)), np.random.default_rng(0))


def test_the_version_changes_with_every_meaningful_setting() -> None:
    base = build("linucb").version
    assert base != build("linucb", alpha=0.9).version
    assert base != build("linucb", propensity_floor=0.1).version
    assert base != build("linucb", ridge=2.0).version
    assert base != build("linucb", discount=0.99).version
    assert base != build("linucb", feature_version="ctx-other").version
    assert base != build("thompson").version
    # Learning does not change the configuration version: the same policy is still the same policy.
    policy = build("linucb")
    policy.update(np.zeros(DIM), 0, 0.5)
    assert policy.version == base
    assert policy.updates == 1


# --- snapshots and the factory ------------------------------------------------------------------


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_a_snapshot_round_trips_through_the_factory(kind: str) -> None:
    policy = build_policy(
        PolicyHyperparameters(kind=kind, discount=0.999),
        arms=ARMS,
        dim=DIM,
        feature_version="ctx-test",
    )
    rng = np.random.default_rng(17)
    for step in range(40):
        policy.update(rng.normal(size=DIM) * 0.3, step % 3, float(rng.random()))
    restored = policy_from_snapshot(policy.snapshot())
    assert restored.version == policy.version
    features = rng.normal(size=DIM) * 0.3
    assert restored.posterior.means(features) == pytest.approx(policy.posterior.means(features))
    assert restored.posterior.widths(features) == pytest.approx(policy.posterior.widths(features))


def test_a_snapshot_whose_parameters_were_edited_is_refused() -> None:
    """The version is a hash of the parameters, so tampering is detectable, not merely unlikely."""
    policy = build("linucb")
    snapshot = policy.snapshot()
    tampered = type(snapshot)(
        name=snapshot.name,
        version=snapshot.version,
        params={**dict(snapshot.params), "alpha": 5.0},
        arrays=snapshot.arrays,
    )
    with pytest.raises(ValueError, match="does not match the rebuilt policy"):
        policy_from_snapshot(tampered)


def test_load_snapshot_refuses_a_different_configuration() -> None:
    snapshot = build("linucb", alpha=0.3).snapshot()
    with pytest.raises(ValueError, match="does not match policy version"):
        build("linucb", alpha=0.9).load_snapshot(snapshot)


def test_policy_from_registry_files_rebuilds_and_verifies() -> None:
    import json

    policy = build("thompson")
    policy.update(np.ones(DIM) * 0.1, 1, 0.7)
    snapshot = policy.snapshot()
    files = {
        "params.json": json.dumps(dict(snapshot.params)).encode("utf-8"),
        "posterior.npz": dumps_npz(dict(snapshot.arrays)),
    }
    restored = policy_from_registry_files(files)
    assert restored.version == policy.version
    with pytest.raises(ValueError, match="missing"):
        policy_from_registry_files({"params.json": files["params.json"]})


def test_registry_files_must_hold_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        policy_from_registry_files(
            {"params.json": b"[1, 2]", "posterior.npz": dumps_npz({"gram": np.zeros((2, 2, 2))})}
        )


def test_snapshot_parameters_must_be_well_typed() -> None:
    policy = build("linucb")
    snapshot = policy.snapshot()
    broken = type(snapshot)(
        name=snapshot.name,
        version=snapshot.version,
        params={**dict(snapshot.params), "dim": "six"},
        arrays=snapshot.arrays,
    )
    with pytest.raises(ValueError, match="must be a number"):
        policy_from_snapshot(broken)
    no_arms = type(snapshot)(
        name=snapshot.name,
        version=snapshot.version,
        params={key: value for key, value in snapshot.params.items() if key != "arms"},
        arrays=snapshot.arrays,
    )
    with pytest.raises(ValueError, match="'arms' must be a list"):
        policy_from_snapshot(no_arms)


def test_an_unknown_policy_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown policy kind"):
        PolicyHyperparameters(kind="bayes_ucb")


# --- non-learning baselines ---------------------------------------------------------------------


def test_static_arm_always_picks_its_target_but_still_logs_propensities() -> None:
    policy = StaticArm(arms=ARMS, dim=DIM, target="opus", propensity_floor=0.02)
    probabilities = policy.probabilities(np.zeros(DIM), np.random.default_rng(0))
    assert int(np.argmax(probabilities)) == 2
    assert float(probabilities.min()) == pytest.approx(0.02)
    assert policy.updates == 0
    policy.update(np.zeros(DIM), 0, 1.0)  # a no-op, by design
    assert policy.updates == 0
    assert policy.name == "static_opus"


def test_static_arm_rejects_an_unknown_target() -> None:
    with pytest.raises(ValueError, match="not one of the arms"):
        StaticArm(arms=ARMS, dim=DIM, target="gpt")


def test_a_fully_deterministic_baseline_logs_zero_propensity_arms() -> None:
    """Which is exactly why the baselines accept a floor: such a log evaluates nothing."""
    policy = StaticArm(arms=ARMS, dim=DIM, target="haiku", propensity_floor=0.0)
    probabilities = policy.probabilities(np.zeros(DIM), np.random.default_rng(0))
    assert probabilities == pytest.approx([1.0, 0.0, 0.0])


def test_the_keyword_rule_reads_the_same_features_the_bandits_see() -> None:
    rule = KeywordRule(
        arms=ARMS, dim=DIM + 40, code_arm="opus", long_arm="sonnet", default_arm="haiku"
    )
    rows = np.zeros((3, DIM + 40))
    rows[0, feature_index("has_code_fence")] = 1.0
    rows[1, feature_index("log_chars")] = 10.0
    chosen = rule.chosen_arms(rows)
    assert list(chosen) == [2, 1, 0]


def test_the_keyword_rule_validates_its_arms() -> None:
    with pytest.raises(ValueError, match="not one of"):
        KeywordRule(arms=ARMS, dim=DIM, code_arm="nope", long_arm="sonnet", default_arm="haiku")
    with pytest.raises(ValueError, match="long_prompt_chars"):
        KeywordRule(
            arms=ARMS,
            dim=DIM,
            code_arm="opus",
            long_arm="sonnet",
            default_arm="haiku",
            long_prompt_chars=0,
        )


def test_static_policies_validate_shapes_and_snapshot() -> None:
    policy = StaticArm(arms=ARMS, dim=DIM, target="haiku")
    with pytest.raises(ValueError, match="shape"):
        policy.probabilities(np.zeros(DIM + 1), np.random.default_rng(0))
    with pytest.raises(ValueError, match="shape"):
        policy.probabilities_matrix(np.zeros((2, DIM + 1)), np.random.default_rng(0))
    snapshot = policy.snapshot()
    assert snapshot.params["target"] == "haiku"
    with pytest.raises(ValueError, match="at least two arms"):
        StaticPolicy(arms=("solo",), dim=DIM)
    with pytest.raises(NotImplementedError):
        StaticPolicy(arms=ARMS, dim=DIM).chosen_arms(np.zeros((1, DIM)))
