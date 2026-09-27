"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.

Beyond the baseline HTTP settings, this file holds the router's own configuration: the four scoped
key lists, the policy and its exploration settings, the reward trade-off, per-tenant budget caps,
feedback limits, breaker thresholds and the artifact directories. Everything that changes what the
router decides is here rather than in code, and the pieces that define the objective (reward
weights, propensity floor, policy hyperparameters) end up inside the policy version, so a logged
decision can always be traced to the configuration that produced it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(default="bandit-llm-router", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- scoped API keys (SHA-256 digests; see `bandit-llm-router generate-api-key`) ---
    route_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    feedback_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    admin_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    approver_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- featurizer ---
    # Keyed BLAKE2b of the prompt is the only prompt-derived value that is stored. Without a key,
    # digests of short prompts are guessable by enumeration (see docs/THREAT_MODEL.md).
    feature_digest_key: SecretStr | None = None
    pricing_path: Path = Path("config/pricing.toml")

    # --- policy ---
    policy_kind: Literal["linucb", "thompson", "epsilon_greedy"] = "linucb"
    policy_alpha: float = Field(default=0.5, ge=0.0, le=10.0)
    policy_epsilon: float = Field(default=0.05, ge=0.0, le=1.0)
    policy_posterior_sd: float = Field(default=0.3, gt=0.0, le=10.0)
    policy_mc_samples: int = Field(default=24, ge=1, le=512)
    policy_ridge: float = Field(default=1.0, gt=0.0)
    policy_discount: float = Field(default=1.0, gt=0.0, le=1.0)
    propensity_floor: float = Field(default=0.02, ge=0.0, le=0.33)

    # --- reward trade-off ---
    reward_cost_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    reward_cost_reference_micros: int = Field(default=14_000, gt=0)

    # --- per-tenant budgets ---
    tenant_budget_usd: float = Field(default=5.0, ge=0.0)
    # "tenant_a:10,tenant_b:2" overrides the default for named tenants.
    tenant_budget_caps: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    tenant_budget_window_seconds: float = Field(default=86_400.0, gt=0.0)

    # --- feedback ---
    feedback_rate_limit_per_second: float = Field(default=20.0, gt=0)
    feedback_rate_limit_burst: int = Field(default=60, ge=1)
    feedback_ttl_seconds: float = Field(default=86_400.0, gt=0.0)
    feedback_max_reports_per_decision: int = Field(default=8, ge=1)

    # --- circuit breaker ---
    breaker_failure_threshold: int = Field(default=3, ge=1)
    breaker_cooldown_seconds: float = Field(default=30.0, gt=0.0)

    # --- screening and tracking ---
    screening_min_ess: float = Field(default=500.0, gt=0.0)
    screening_horizon_requests: int = Field(default=100_000, ge=1)
    screening_max_exploration_usd: float = Field(default=250.0, gt=0.0)
    # The service screens on the log its own *learning* policy wrote, so the rows are not i.i.d.
    # and a bootstrap over single rows would be assuming they are. Resample contiguous blocks
    # instead; see ope/estimators.py. Set to 1 to opt back into the plain row bootstrap.
    screening_bootstrap_block_size: int = Field(default=250, ge=1)
    mlflow_dir: Path = Path("var/mlflow")

    # --- live provider ---
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # The SDK default. Non-streaming requests up to 16K output tokens can legitimately take minutes;
    # a shorter total timeout cuts them off, and each retry can be billed again.
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    llm_refusal_fallback: bool = True
    budget_max_steps: int = Field(default=50, ge=0)
    budget_max_tokens: int = Field(default=200_000, ge=0)
    budget_max_usd: float = Field(default=1.0, ge=0)

    # --- storage ---
    data_dir: Path = Path("var")

    @field_validator(
        "api_key_sha256",
        "route_api_key_sha256",
        "feedback_api_key_sha256",
        "admin_api_key_sha256",
        "approver_api_key_sha256",
        mode="before",
    )
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator(
        "api_key_sha256",
        "route_api_key_sha256",
        "feedback_api_key_sha256",
        "admin_api_key_sha256",
        "approver_api_key_sha256",
    )
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("API key entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("tenant_budget_caps", mode="before")
    @classmethod
    def _parse_budget_caps(cls, value: object) -> object:
        """Parse ``tenant_a:10,tenant_b:2.5`` into a mapping."""
        if not isinstance(value, str):
            return value
        caps: dict[str, float] = {}
        for item in value.split(","):
            entry = item.strip()
            if not entry:
                continue
            tenant, separator, amount = entry.partition(":")
            if not separator:
                raise ValueError("tenant_budget_caps entries must look like tenant:usd")
            try:
                caps[tenant.strip()] = float(amount)
            except ValueError:
                raise ValueError(f"tenant_budget_caps: {amount!r} is not a number") from None
        return caps

    @field_validator("tenant_budget_caps")
    @classmethod
    def _check_budget_caps(cls, value: dict[str, float]) -> dict[str, float]:
        for tenant, amount in value.items():
            if not tenant:
                raise ValueError("tenant_budget_caps has an empty tenant name")
            if amount < 0.0:
                raise ValueError(f"tenant_budget_caps[{tenant}] must be non-negative")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
