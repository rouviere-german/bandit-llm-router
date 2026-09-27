"""Which provider serves a routed request, and how an arm becomes a model.

Offline is the default and it must stay that way: with no ``ANTHROPIC_API_KEY`` the whole repo runs
on a seeded fake, so the tests, the demo and the benches never spend money and never need the
network. The live provider is only constructed when a key is present, and the router never names a
model -- it picks an arm, and the pricing config maps that arm to exactly one pinned model.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from bandit_llm_router.agentic.providers.anthropic import AnthropicProvider
from bandit_llm_router.agentic.providers.base import ChatMessage, LLMRequest, TextBlock
from bandit_llm_router.agentic.providers.fake import FakeProvider
from bandit_llm_router.arms import ArmCatalog
from bandit_llm_router.core.config import Settings
from bandit_llm_router.provider import OfflineRouterPolicy, build_provider, offline_provider

pytestmark = pytest.mark.unit


def request_for(
    arm: str, *, prompt: str = "Summarize these notes.", max_tokens: int = 512
) -> LLMRequest:
    return LLMRequest(messages=(ChatMessage.user(prompt),), role=arm, max_tokens=max_tokens)


def test_with_no_key_the_offline_provider_is_used(catalog: ArmCatalog) -> None:
    provider = build_provider(Settings(_env_file=None, environment="test"), catalog)
    assert isinstance(provider, FakeProvider)


def test_an_empty_key_still_counts_as_no_key(catalog: ArmCatalog) -> None:
    settings = Settings(_env_file=None, environment="test", anthropic_api_key=SecretStr(""))
    assert isinstance(build_provider(settings, catalog), FakeProvider)


def test_a_configured_key_builds_the_live_provider(catalog: ArmCatalog) -> None:
    """Construction only: nothing is sent, and the SDK is imported lazily inside the provider."""
    settings = Settings(
        _env_file=None, environment="test", anthropic_api_key=SecretStr("sk-ant-test-key")
    )
    provider = build_provider(settings, catalog)
    assert isinstance(provider, AnthropicProvider)
    # Every arm maps to exactly one pinned model, and the provider knows the mapping.
    assert provider.model_for("opus") == "claude-opus-5"
    assert provider.model_for("haiku") == "claude-haiku-4-5"


def test_the_offline_provider_answers_per_arm(catalog: ArmCatalog) -> None:
    provider = offline_provider(catalog, seed=3)
    for arm in catalog.names:
        response = provider.complete(request_for(arm))
        assert response.model == catalog.spec(arm).model
        assert response.stop_reason == "end_turn"
        assert f"[offline:{arm}]" in response.text
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


def test_the_offline_reply_length_follows_the_arms_own_estimate(catalog: ArmCatalog) -> None:
    """So the cost the offline path commits is in the range the live path would bill."""
    provider = offline_provider(catalog)
    for arm in catalog.names:
        spec = catalog.spec(arm)
        response = provider.complete(request_for(arm, max_tokens=8_192))
        assert response.usage.output_tokens == spec.expected_output_tokens


def test_the_offline_reply_respects_the_requested_ceiling(catalog: ArmCatalog) -> None:
    provider = offline_provider(catalog)
    response = provider.complete(request_for("opus", max_tokens=64))
    assert response.usage.output_tokens == 64


def test_the_offline_provider_is_deterministic(catalog: ArmCatalog) -> None:
    first = offline_provider(catalog, seed=7).complete(request_for("sonnet"))
    second = offline_provider(catalog, seed=7).complete(request_for("sonnet"))
    assert first.text == second.text
    assert first.usage == second.usage


def test_an_unknown_role_falls_back_to_a_fake_model(catalog: ArmCatalog) -> None:
    policy = OfflineRouterPolicy(catalog)
    import random

    response = policy(request_for("default"), random.Random(0), 0)
    assert response.model == "fake-model"
    assert response.usage.output_tokens == 256


def test_input_tokens_scale_with_the_prompt(catalog: ArmCatalog) -> None:
    provider = offline_provider(catalog)
    short = provider.complete(request_for("haiku", prompt="hi"))
    long = provider.complete(request_for("haiku", prompt="word " * 500))
    assert long.usage.input_tokens > short.usage.input_tokens


def test_only_text_blocks_count_toward_the_prompt_size(catalog: ArmCatalog) -> None:
    from bandit_llm_router.agentic.providers.base import ToolResultBlock

    policy = OfflineRouterPolicy(catalog)
    import random

    mixed = LLMRequest(
        messages=(
            ChatMessage(
                role="user",
                content=(
                    TextBlock(text="abcd"),
                    ToolResultBlock(tool_use_id="t1", content="x" * 400),
                ),
            ),
        ),
        role="haiku",
        max_tokens=128,
    )
    text_only = LLMRequest(
        messages=(ChatMessage(role="user", content=(TextBlock(text="abcd"),)),),
        role="haiku",
        max_tokens=128,
    )
    assert (
        policy(mixed, random.Random(0), 0).usage.input_tokens
        == policy(text_only, random.Random(0), 0).usage.input_tokens
    )
