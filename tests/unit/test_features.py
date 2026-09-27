"""The featurizer and its data contract.

Three properties matter and all three are enforced in code rather than assumed:

* **Determinism.** A logged feature row is only replayable months later if the same context produces
  the same vector in any process. So the featurizer is hash-based with no fitted vocabulary, and the
  test checks a fresh instance reproduces another one's output exactly.
* **Boundedness.** Every feature satisfies ``|x| <= FEATURE_ABS_MAX``. An unbounded feature would
  destabilize the ridge posteriors and inflate importance weights without anything looking broken.
* **No prompt leaves.** What is logged next to the features is a keyed digest, not text. The digest
  is keyed so short prompts are not trivially enumerable.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bandit_llm_router.features import (
    DENSE_FEATURE_NAMES,
    FEATURE_ABS_MAX,
    FEATURE_DIM,
    HASH_BUCKETS,
    MAX_PROMPT_CHARS,
    FeatureContractError,
    Featurizer,
    RoutingContext,
    validate_feature_matrix,
    validate_features,
)

pytestmark = pytest.mark.unit

prompts = st.text(min_size=1, max_size=400)


def context(prompt: str, **overrides: object) -> RoutingContext:
    values: dict[str, object] = {"tenant_id": "tenant_a", "prompt": prompt}
    values.update(overrides)
    return RoutingContext.model_validate(values)


# --- the request contract ------------------------------------------------------------------------


def test_the_routing_context_bounds_everything_a_caller_controls() -> None:
    context("hello")
    with pytest.raises(ValidationError):
        context("")
    with pytest.raises(ValidationError):
        context("x" * (MAX_PROMPT_CHARS + 1))
    with pytest.raises(ValidationError):
        context("hi", max_output_tokens=0)
    with pytest.raises(ValidationError):
        context("hi", max_output_tokens=100_000)
    with pytest.raises(ValidationError):
        context("hi", tier="urgent")
    with pytest.raises(ValidationError):
        context("hi", tenant_id="Tenant With Spaces")
    with pytest.raises(ValidationError):
        RoutingContext(tenant_id="tenant_a", prompt="hi", surprise=1)  # type: ignore[call-arg]


def test_the_context_is_frozen() -> None:
    ctx = context("hello")
    with pytest.raises(ValueError, match="frozen"):
        ctx.prompt = "other"  # type: ignore[misc]


@given(prompts)
def test_the_reservation_bound_is_never_below_the_estimate(prompt: str) -> None:
    """Reservations must over-count tokens, or the budget cap stops being a bound."""
    ctx = context(prompt)
    assert ctx.input_token_ceiling >= ctx.estimated_input_tokens >= 1


def test_the_token_estimates_use_the_documented_ratios() -> None:
    ctx = context("x" * 400)
    assert ctx.estimated_input_tokens == 100
    assert ctx.input_token_ceiling == 200


# --- the feature contract -----------------------------------------------------------------------


def test_the_contract_rejects_anything_a_policy_must_not_see() -> None:
    good = np.zeros(FEATURE_DIM)
    assert validate_features(good) is good
    with pytest.raises(FeatureContractError, match="shape"):
        validate_features(np.zeros(FEATURE_DIM - 1))
    with pytest.raises(FeatureContractError, match="shape"):
        validate_features(np.zeros((1, FEATURE_DIM)))
    with pytest.raises(FeatureContractError, match="float64"):
        validate_features(np.zeros(FEATURE_DIM, dtype=np.float32))
    nan = np.zeros(FEATURE_DIM)
    nan[0] = np.nan
    with pytest.raises(FeatureContractError, match="NaN"):
        validate_features(nan)
    big = np.zeros(FEATURE_DIM)
    big[0] = FEATURE_ABS_MAX + 1.0
    with pytest.raises(FeatureContractError, match="bound"):
        validate_features(big)


def test_the_matrix_contract_mirrors_the_row_contract() -> None:
    assert validate_feature_matrix(np.zeros((3, FEATURE_DIM))).shape == (3, FEATURE_DIM)
    assert validate_feature_matrix(np.zeros((0, FEATURE_DIM))).shape == (0, FEATURE_DIM)
    with pytest.raises(FeatureContractError, match="columns"):
        validate_feature_matrix(np.zeros((3, FEATURE_DIM + 1)))
    with pytest.raises(FeatureContractError, match="NaN"):
        validate_feature_matrix(np.full((3, FEATURE_DIM), np.inf))
    with pytest.raises(FeatureContractError, match="bound"):
        validate_feature_matrix(np.full((3, FEATURE_DIM), FEATURE_ABS_MAX + 1.0))


def test_the_dimension_is_the_sum_of_its_documented_parts() -> None:
    assert HASH_BUCKETS + len(DENSE_FEATURE_NAMES) + 1 == FEATURE_DIM


# --- the featurizer -----------------------------------------------------------------------------


@given(prompts)
def test_every_vector_satisfies_the_contract(prompt: str) -> None:
    featurizer = Featurizer()
    result = featurizer.transform(context(prompt))
    assert result.vector.shape == (featurizer.dim,)
    assert float(np.max(np.abs(result.vector))) <= FEATURE_ABS_MAX
    assert np.all(np.isfinite(result.vector))
    assert result.vector[-1] == 1.0  # the bias term


@given(prompts)
def test_two_featurizers_agree_exactly(prompt: str) -> None:
    """No fitted state, so a log written by one process is replayable by another."""
    first = Featurizer().transform(context(prompt))
    second = Featurizer().transform(context(prompt))
    assert first.vector == pytest.approx(second.vector)
    assert first.context_digest == second.context_digest


def test_the_names_line_up_with_the_vector(featurizer: Featurizer) -> None:
    names = featurizer.feature_names()
    assert len(names) == featurizer.dim
    assert names[-1] == "bias"
    assert names[HASH_BUCKETS] == DENSE_FEATURE_NAMES[0]


def test_the_dense_features_read_the_structure_a_router_can_use(featurizer: Featurizer) -> None:
    def dense(prompt: str, name: str, **overrides: object) -> float:
        vector = featurizer.transform(context(prompt, **overrides)).vector
        return float(vector[HASH_BUCKETS + DENSE_FEATURE_NAMES.index(name)])

    assert dense("```python\nx = 1\n```", "has_code_fence") == 1.0
    assert dense("plain prose", "has_code_fence") == 0.0
    assert dense("what is this?", "has_question") == 1.0
    assert dense("a" * 1_000, "log_chars") > dense("a" * 10, "log_chars")
    assert dense("123456", "digit_ratio") == 1.0
    assert dense("ABCDEF", "uppercase_ratio") == 1.0
    assert dense("hi", "is_priority_tier", tier="priority") == 1.0
    assert dense("hi", "is_priority_tier") == 0.0
    assert dense("a\n" * 60, "newline_density") == 1.0
    assert dense("one one one", "distinct_word_ratio") == pytest.approx(1 / 3)
    assert dense("hi", "log_max_output_tokens", max_output_tokens=4_000) > dense(
        "hi", "log_max_output_tokens", max_output_tokens=32
    )


def test_different_prompts_produce_different_vectors(featurizer: Featurizer) -> None:
    first = featurizer.transform(context("Extract every field from this invoice record."))
    second = featurizer.transform(context("Hi, quick question: how are you?"))
    assert not np.allclose(first.vector, second.vector)


# --- the digest ---------------------------------------------------------------------------------


@given(prompts)
def test_the_digest_is_short_hex_and_prompt_specific(prompt: str) -> None:
    featurizer = Featurizer()
    digest = featurizer.digest(prompt)
    assert len(digest) == 32
    assert int(digest, 16) >= 0
    assert featurizer.digest(prompt + "!") != digest


def test_the_key_changes_every_digest_so_prompts_are_not_enumerable() -> None:
    unkeyed = Featurizer()
    keyed = Featurizer(digest_key=b"deployment-secret")
    other = Featurizer(digest_key=b"another-secret")
    assert unkeyed.digest("yes") != keyed.digest("yes")
    assert keyed.digest("yes") != other.digest("yes")
    # The key is a deployment secret, not part of the feature version.
    assert unkeyed.version == keyed.version


def test_an_over_long_digest_key_is_refused() -> None:
    with pytest.raises(ValueError, match="too long"):
        Featurizer(digest_key=b"x" * 100)


# --- the version and the cache -------------------------------------------------------------------


def test_the_version_pins_the_configuration_that_produced_a_log() -> None:
    base = Featurizer()
    assert base.version.startswith("ctx-v1-")
    assert Featurizer().version == base.version
    assert Featurizer(buckets=32).version != base.version
    assert Featurizer(ngram_range=(2, 4)).version != base.version
    assert Featurizer(buckets=32).dim == 32 + len(DENSE_FEATURE_NAMES) + 1


def test_the_cache_returns_the_same_immutable_vector(featurizer: Featurizer) -> None:
    first = featurizer.transform(context("cached prompt"))
    second = featurizer.transform(context("cached prompt"))
    assert second.vector is first.vector
    with pytest.raises(ValueError, match="read-only"):
        first.vector[0] = 5.0


def test_the_cache_keys_on_the_whole_context_not_just_the_prompt(featurizer: Featurizer) -> None:
    standard = featurizer.transform(context("same text"))
    priority = featurizer.transform(context("same text", tier="priority"))
    assert standard.context_digest == priority.context_digest
    assert not np.allclose(standard.vector, priority.vector)


def test_the_cache_evicts_the_least_recently_used_entry() -> None:
    featurizer = Featurizer(cache_size=2)
    first = featurizer.transform(context("one")).vector
    featurizer.transform(context("two"))
    featurizer.transform(context("three"))  # evicts "one"
    assert featurizer.transform(context("one")).vector is not first


def test_the_featurizer_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="buckets"):
        Featurizer(buckets=1)
    with pytest.raises(ValueError, match="ngram_range"):
        Featurizer(ngram_range=(5, 3))
    with pytest.raises(ValueError, match="cache_size"):
        Featurizer(cache_size=0)


# --- batches ------------------------------------------------------------------------------------


def test_a_batch_matches_the_per_row_transform(featurizer: Featurizer) -> None:
    contexts = [
        context("Extract the fields:\norder_id: 55\namount_eur: 12"),
        context("Hey -- quick one, thanks!", tier="priority"),
        context("```python\ntotal = compute(3, rows)\n```", max_output_tokens=1_024),
    ]
    matrix = featurizer.transform_batch(contexts)
    assert matrix.shape == (3, featurizer.dim)
    for row, ctx in enumerate(contexts):
        assert matrix[row] == pytest.approx(featurizer.transform(ctx).vector)


def test_an_empty_batch_has_the_right_shape(featurizer: Featurizer) -> None:
    assert featurizer.transform_batch([]).shape == (0, featurizer.dim)
