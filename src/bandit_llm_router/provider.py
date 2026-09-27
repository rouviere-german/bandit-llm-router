"""Which provider serves a routed request, and how an arm becomes a model.

The router never names a model: it picks an *arm*, and the provider maps that arm to exactly one
pinned model ID. In SDK terms the arm is the request's ``role``, and
:meth:`ArmCatalog.role_models` is the role-to-model mapping ``AnthropicProvider`` is built with.
Adding a model therefore means editing ``config/pricing.toml`` (whose models are checked against the
allowlist in :mod:`bandit_llm_router.arms`), not editing a call site.

Offline is the default. With no ``ANTHROPIC_API_KEY``, :func:`build_provider` returns a
:class:`~bandit_llm_router.agentic.providers.fake.FakeProvider` driven by a seeded policy that
answers deterministically and reports token counts scaled to the arm's expected output length, so
costs, budgets and metrics all exercise realistic numbers with no network and no spend. The whole
test suite, the demo and the benches run on it. The live provider is constructed only when a key is
present, and the SDK is imported lazily inside it, so an offline process never even loads it.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from bandit_llm_router.agentic.providers.anthropic import AnthropicProvider
from bandit_llm_router.agentic.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TextBlock,
    Usage,
)
from bandit_llm_router.agentic.providers.fake import FakeProvider, estimate_tokens
from bandit_llm_router.arms import ArmCatalog

if TYPE_CHECKING:
    from bandit_llm_router.core.config import Settings

__all__ = ["OfflineRouterPolicy", "build_provider", "offline_provider"]

_WORDS_PER_TOKEN_CHARS = 4


class OfflineRouterPolicy:
    """Deterministic stand-in for a model call, with per-arm token accounting.

    The reply length follows the arm's ``expected_output_tokens``, so the cost the offline path
    commits is in the same range the live path would bill. Nothing here is a claim about model
    quality: the reward signal comes from feedback, and offline that means the simulator's rubric.
    """

    def __init__(self, catalog: ArmCatalog) -> None:
        self._catalog = catalog

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del rng, turn
        spec = self._catalog.spec(request.role) if request.role in self._catalog.names else None
        model = spec.model if spec is not None else "fake-model"
        target_tokens = min(
            request.max_tokens, spec.expected_output_tokens if spec is not None else 256
        )
        text = f"[offline:{request.role}] " + "answer " * max(
            1, (target_tokens * _WORDS_PER_TOKEN_CHARS) // len("answer ")
        )
        prompt_chars = sum(
            len(block.text)
            for message in request.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        return LLMResponse(
            model=model,
            stop_reason="end_turn",
            text=text,
            usage=Usage(
                input_tokens=estimate_tokens("x" * max(1, prompt_chars)),
                output_tokens=target_tokens,
            ),
        )


def offline_provider(catalog: ArmCatalog, *, seed: int = 0) -> FakeProvider:
    """A seeded offline provider for tests, the demo and the benches."""
    return FakeProvider(OfflineRouterPolicy(catalog), seed=seed)


def build_provider(settings: Settings, catalog: ArmCatalog, *, seed: int = 0) -> LLMProvider:
    """The live provider when a key is configured, otherwise the offline one."""
    key = settings.anthropic_api_key
    if key is None or not key.get_secret_value():
        return offline_provider(catalog, seed=seed)
    return AnthropicProvider.from_settings(settings, role_models=catalog.role_models())
