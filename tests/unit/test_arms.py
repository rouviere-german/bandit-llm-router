"""Routing arms: the pinned allowlist, and cost arithmetic that is never optimistic.

Two claims live here. The allowlist is a security control -- a pricing file naming an unreviewed
model must be refused at load time rather than quietly routed to. And the cost estimate must round
*up*, because the tenant budget guard reserves the ceiling before a call and a ceiling that
under-counts turns the cap into a suggestion.
"""

from __future__ import annotations

import tomllib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bandit_llm_router.agentic.pricing import PRICES_PER_MTOK
from bandit_llm_router.arms import (
    PINNED_MODELS,
    ArmCatalog,
    ArmConfigError,
    ArmSpec,
    load_catalog,
)

pytestmark = pytest.mark.unit


def spec(name: str, model: str, **overrides: object) -> ArmSpec:
    values: dict[str, object] = {
        "name": name,
        "model": model,
        "input_usd_per_mtok": Decimal("1.00"),
        "output_usd_per_mtok": Decimal("5.00"),
        "expected_output_tokens": 200,
        "max_output_tokens": 1_024,
        "fallback_priority": 1,
    }
    values.update(overrides)
    return ArmSpec.model_validate(values)


# --- the allowlist -------------------------------------------------------------------------------


def test_an_unreviewed_model_cannot_become_an_arm() -> None:
    with pytest.raises(ValueError, match="pinned allowlist"):
        spec("rogue", "claude-experimental-9")


def test_every_pinned_model_is_a_claude_model_this_repo_prices() -> None:
    assert {"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"} == PINNED_MODELS
    for model in PINNED_MODELS:
        assert model in PRICES_PER_MTOK


def test_the_shipped_pricing_file_agrees_with_the_sdk_price_table(repo_root: Path) -> None:
    """One source of truth for money: the router's config and the LLM ledger must not disagree."""
    catalog = load_catalog(repo_root / "config" / "pricing.toml")
    for arm in catalog.specs:
        input_price, output_price = PRICES_PER_MTOK[arm.model]
        assert arm.input_usd_per_mtok == input_price
        assert arm.output_usd_per_mtok == output_price


def test_the_shipped_pricing_file_describes_three_ordered_arms(catalog: ArmCatalog) -> None:
    assert catalog.names == ("haiku", "sonnet", "opus")
    assert catalog.models == ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
    assert len(catalog) == 3
    assert catalog.fallback_order == (0, 1, 2)
    assert catalog.role_models()["opus"] == "claude-opus-5"
    assert [arm.name for arm in catalog.specs] == ["haiku", "sonnet", "opus"]


def test_a_catalog_refuses_degenerate_arm_sets() -> None:
    with pytest.raises(ArmConfigError, match="at least two arms"):
        ArmCatalog([spec("haiku", "claude-haiku-4-5")])
    with pytest.raises(ArmConfigError, match="unique"):
        ArmCatalog([spec("dup", "claude-haiku-4-5"), spec("dup", "claude-sonnet-5")])
    with pytest.raises(ArmConfigError, match="at most one arm"):
        ArmCatalog([spec("a", "claude-haiku-4-5"), spec("b", "claude-haiku-4-5")])


def test_an_arm_cannot_be_priced_at_zero() -> None:
    with pytest.raises(ValidationError):
        spec("free", "claude-haiku-4-5", input_usd_per_mtok=Decimal("0"))


def test_an_expected_length_above_the_arm_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot exceed max_output_tokens"):
        spec("odd", "claude-haiku-4-5", expected_output_tokens=2_000, max_output_tokens=1_000)


def test_an_unknown_arm_name_is_a_config_error(catalog: ArmCatalog) -> None:
    with pytest.raises(ArmConfigError, match="unknown arm"):
        catalog.index("gpt")


def test_the_fallback_order_follows_the_configured_priority() -> None:
    reversed_priority = ArmCatalog(
        [
            spec("expensive", "claude-opus-5", fallback_priority=9),
            spec("cheap", "claude-haiku-4-5", fallback_priority=1),
            spec("mid", "claude-sonnet-5", fallback_priority=5),
        ]
    )
    assert reversed_priority.fallback_order == (1, 2, 0)


# --- loading -------------------------------------------------------------------------------------


def test_a_missing_pricing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ArmConfigError, match="not found"):
        load_catalog(tmp_path / "absent.toml")


def test_malformed_toml_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "pricing.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ArmConfigError, match="not valid TOML"):
        load_catalog(path)


def test_an_unknown_key_in_the_pricing_file_is_refused(tmp_path: Path, repo_root: Path) -> None:
    source = (repo_root / "config" / "pricing.toml").read_text(encoding="utf-8")
    path = tmp_path / "pricing.toml"
    path.write_text(source + '\nsurprise = "value"\n', encoding="utf-8")
    with pytest.raises(ArmConfigError, match="is invalid"):
        load_catalog(path)


def test_a_single_arm_pricing_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pricing.toml"
    path.write_text(
        "schema_version = 1\n"
        "[[arm]]\n"
        'name = "haiku"\n'
        'model = "claude-haiku-4-5"\n'
        "input_usd_per_mtok = 1.0\n"
        "output_usd_per_mtok = 5.0\n"
        "expected_output_tokens = 200\n"
        "max_output_tokens = 1024\n"
        "fallback_priority = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ArmConfigError, match="is invalid"):
        load_catalog(path)


def test_the_pricing_file_declares_its_schema_version(repo_root: Path) -> None:
    raw = tomllib.loads((repo_root / "config" / "pricing.toml").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1


# --- cost arithmetic -----------------------------------------------------------------------------


def test_cost_is_priced_per_token_and_rounded_up() -> None:
    arm = spec(
        "haiku",
        "claude-haiku-4-5",
        input_usd_per_mtok=Decimal("1.00"),
        output_usd_per_mtok=Decimal("5.00"),
    )
    # USD per million tokens is numerically micro-USD per token.
    assert arm.cost_micros(1_000, 100) == 1_000 * 1 + 100 * 5
    fractional = spec("x", "claude-sonnet-5", input_usd_per_mtok=Decimal("0.5"))
    assert fractional.cost_micros(1, 0) == 1  # 0.5 rounds up, never down


def test_negative_token_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        spec("haiku", "claude-haiku-4-5").cost_micros(-1, 0)


@given(
    input_tokens=st.integers(min_value=0, max_value=100_000),
    requested=st.integers(min_value=1, max_value=8_192),
)
def test_the_ceiling_bounds_the_expected_cost_when_the_request_is_large_enough(
    catalog: ArmCatalog, input_tokens: int, requested: int
) -> None:
    for index, arm in enumerate(catalog.specs):
        expected = catalog.expected_cost_micros(index, input_tokens)
        ceiling = catalog.cost_ceiling_micros(index, input_tokens, requested)
        capped = min(requested, arm.max_output_tokens)
        if capped >= arm.expected_output_tokens:
            assert ceiling >= expected


def test_the_ceiling_never_charges_more_than_the_arms_own_output_cap(catalog: ArmCatalog) -> None:
    arm = catalog.spec("opus")
    huge = catalog.cost_ceiling_micros("opus", 100, 1_000_000)
    assert huge == arm.cost_micros(100, arm.max_output_tokens)


def test_a_zero_length_request_is_refused(catalog: ArmCatalog) -> None:
    with pytest.raises(ValueError, match="max_output_tokens must be"):
        catalog.cost_ceiling_micros("haiku", 100, 0)


def test_the_vectorized_cost_matrices_match_the_per_call_methods(catalog: ArmCatalog) -> None:
    """The benches and the simulator use the matrices; the service uses the scalars. One formula."""
    rng = np.random.default_rng(0)
    input_tokens = rng.integers(1, 4_000, size=25)
    requested = rng.integers(64, 4_000, size=25)
    expected, ceiling = catalog.cost_matrices(input_tokens, requested)
    assert expected.shape == (25, len(catalog))
    for row, (tokens, out) in enumerate(zip(input_tokens, requested, strict=True)):
        for index in range(len(catalog)):
            assert expected[row, index] == pytest.approx(
                catalog.expected_cost_micros(index, int(tokens))
            )
            assert ceiling[row, index] == pytest.approx(
                catalog.cost_ceiling_micros(index, int(tokens), int(out))
            )


def test_the_cost_vectors_match_the_matrices(catalog: ArmCatalog) -> None:
    expected, ceiling = catalog.cost_matrices([500], [700])
    assert catalog.expected_cost_vector(500) == pytest.approx(expected[0])
    assert catalog.cost_ceiling_vector(500, 700) == pytest.approx(ceiling[0])


def test_the_cost_matrices_validate_their_inputs(catalog: ArmCatalog) -> None:
    with pytest.raises(ValueError, match="same shape"):
        catalog.cost_matrices([1, 2], [1])
    with pytest.raises(ValueError, match="non-negative"):
        catalog.cost_matrices([-1], [10])
    with pytest.raises(ValueError, match="at least 1"):
        catalog.cost_matrices([1], [0])


def test_a_bigger_model_costs_more_for_the_same_request(catalog: ArmCatalog) -> None:
    """The premise of the whole repo: routing is a trade-off, so the prices must actually differ."""
    costs = [catalog.expected_cost_micros(name, 1_000) for name in catalog.names]
    assert costs == sorted(costs)
    assert costs[-1] > costs[0] * 2


def test_the_digest_payload_captures_prices_and_limits(catalog: ArmCatalog) -> None:
    payload = catalog.digest_payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    assert len(arms) == 3
    first = arms[0]
    assert isinstance(first, dict)
    assert first["model"] == "claude-haiku-4-5"
    # Prices are stringified so the digest is exact rather than float-dependent.
    assert isinstance(first["input_usd_per_mtok"], str)
