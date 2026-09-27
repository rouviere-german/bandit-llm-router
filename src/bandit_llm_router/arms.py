"""Routing arms: a pinned model allowlist and the pricing config that drives cost estimates.

An arm is a logical routing target (``haiku``, ``sonnet``, ``opus``) bound to exactly one Claude
model ID from :data:`PINNED_MODELS`. The catalog is built from ``config/pricing.toml``; a config
that names a model outside the allowlist, repeats an arm, or prices an arm at zero is refused at
load time, so an unreviewed model can never become routable and no code path can invent one.

Costs are computed in micro-USD with :class:`~decimal.Decimal` prices (USD per million tokens is
numerically equal to micro-USD per token) and rounded up, so an estimate is never optimistic. Two
numbers matter for different reasons:

* :meth:`ArmCatalog.expected_cost_micros` uses the arm's planning estimate of output length and
  feeds the reward's cost penalty and the reported spend.
* :meth:`ArmCatalog.cost_ceiling_micros` uses the request's ``max_output_tokens`` (capped by the
  arm's own limit) and is what the tenant budget guard reserves. A committed cost can never exceed
  the ceiling, which is what makes the per-tenant cap an exact invariant rather than a best effort.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = [
    "DEFAULT_PRICING_PATH",
    "PINNED_MODELS",
    "ArmCatalog",
    "ArmConfigError",
    "ArmSpec",
    "load_catalog",
]

PINNED_MODELS: frozenset[str] = frozenset({"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"})
DEFAULT_PRICING_PATH = Path("config/pricing.toml")
_MICROS_PER_USD = 1_000_000
ArmName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
FloatArray = npt.NDArray[np.float64]


class ArmConfigError(ValueError):
    """The pricing config does not describe a usable set of arms."""


class ArmSpec(BaseModel):
    """One routable arm: a pinned model, its prices and its output-length limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ArmName
    model: str
    input_usd_per_mtok: Decimal = Field(gt=0, le=1000)
    output_usd_per_mtok: Decimal = Field(gt=0, le=1000)
    expected_output_tokens: int = Field(ge=1, le=200_000)
    max_output_tokens: int = Field(ge=1, le=200_000)
    fallback_priority: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def _check_model_and_limits(self) -> ArmSpec:
        if self.model not in PINNED_MODELS:
            raise ValueError(
                f"model {self.model!r} is not on the pinned allowlist {sorted(PINNED_MODELS)}"
            )
        if self.expected_output_tokens > self.max_output_tokens:
            raise ValueError("expected_output_tokens cannot exceed max_output_tokens")
        return self

    def cost_micros(self, input_tokens: int, output_tokens: int) -> int:
        """Cost of one call in micro-USD, rounded up."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        total = input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        return math.ceil(total)


class _PricingFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    arm: tuple[ArmSpec, ...] = Field(min_length=2)


class ArmCatalog:
    """An ordered, immutable set of arms. Arm order is the action index order everywhere."""

    __slots__ = ("_by_name", "_fallback_order", "_specs")

    def __init__(self, specs: Sequence[ArmSpec]) -> None:
        if len(specs) < 2:
            raise ArmConfigError("at least two arms are required for routing to be a decision")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ArmConfigError("arm names must be unique")
        models = [spec.model for spec in specs]
        if len(set(models)) != len(models):
            raise ArmConfigError("each pinned model may back at most one arm")
        self._specs: tuple[ArmSpec, ...] = tuple(specs)
        self._by_name: dict[str, int] = {spec.name: index for index, spec in enumerate(specs)}
        self._fallback_order: tuple[int, ...] = tuple(
            index
            for index, _ in sorted(
                enumerate(specs), key=lambda item: (item[1].fallback_priority, item[0])
            )
        )

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterable[ArmSpec]:
        return iter(self._specs)

    @property
    def specs(self) -> tuple[ArmSpec, ...]:
        return self._specs

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(spec.model for spec in self._specs)

    @property
    def fallback_order(self) -> tuple[int, ...]:
        """Arm indices ordered by ``fallback_priority`` (cheapest-first by convention)."""
        return self._fallback_order

    def role_models(self) -> dict[str, str]:
        """Arm name -> model ID, the mapping ``AnthropicProvider`` uses for request roles."""
        return {spec.name: spec.model for spec in self._specs}

    def index(self, name: str) -> int:
        try:
            return self._by_name[name]
        except KeyError:
            raise ArmConfigError(f"unknown arm {name!r}") from None

    def spec(self, arm: str | int) -> ArmSpec:
        return self._specs[arm if isinstance(arm, int) else self.index(arm)]

    def expected_cost_micros(self, arm: str | int, input_tokens: int) -> int:
        """Planning cost estimate: the arm's expected output length at its own prices."""
        spec = self.spec(arm)
        return spec.cost_micros(input_tokens, spec.expected_output_tokens)

    def cost_ceiling_micros(self, arm: str | int, input_tokens: int, max_output_tokens: int) -> int:
        """Upper bound on what one call can cost, used for budget reservations."""
        spec = self.spec(arm)
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        return spec.cost_micros(input_tokens, min(max_output_tokens, spec.max_output_tokens))

    def expected_cost_vector(self, input_tokens: int) -> FloatArray:
        """Expected cost of every arm in micro-USD, as an array indexed like the arms."""
        return np.asarray(
            [self.expected_cost_micros(index, input_tokens) for index in range(len(self._specs))],
            dtype=np.float64,
        )

    def cost_ceiling_vector(self, input_tokens: int, max_output_tokens: int) -> FloatArray:
        return np.asarray(
            [
                self.cost_ceiling_micros(index, input_tokens, max_output_tokens)
                for index in range(len(self._specs))
            ],
            dtype=np.float64,
        )

    def cost_matrices(
        self, input_tokens: npt.ArrayLike, max_output_tokens: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Expected and worst-case cost of every (request, arm) pair, shape ``(n, n_arms)``.

        Same formula as :meth:`expected_cost_micros` and :meth:`cost_ceiling_micros`, vectorized for
        whole logs and for the simulator; the per-call methods stay the single definition of the
        rounding.
        """
        inputs = np.asarray(input_tokens, dtype=np.float64).ravel()
        outputs = np.asarray(max_output_tokens, dtype=np.float64).ravel()
        if inputs.shape != outputs.shape:
            raise ValueError("input_tokens and max_output_tokens must have the same shape")
        if np.any(inputs < 0) or np.any(outputs < 1):
            raise ValueError("input tokens must be non-negative and output tokens at least 1")
        input_price = np.asarray([float(s.input_usd_per_mtok) for s in self._specs])
        output_price = np.asarray([float(s.output_usd_per_mtok) for s in self._specs])
        expected_out = np.asarray([s.expected_output_tokens for s in self._specs], dtype=np.float64)
        arm_cap = np.asarray([s.max_output_tokens for s in self._specs], dtype=np.float64)
        input_cost = np.outer(inputs, input_price)
        expected = np.ceil(input_cost + expected_out[np.newaxis, :] * output_price[np.newaxis, :])
        capped = np.minimum(outputs[:, np.newaxis], arm_cap[np.newaxis, :])
        ceiling = np.ceil(input_cost + capped * output_price[np.newaxis, :])
        return expected, ceiling

    def digest_payload(self) -> dict[str, object]:
        """Canonical, JSON-safe description of the catalog (part of a policy version)."""
        return {
            "arms": [
                {
                    "name": spec.name,
                    "model": spec.model,
                    "input_usd_per_mtok": str(spec.input_usd_per_mtok),
                    "output_usd_per_mtok": str(spec.output_usd_per_mtok),
                    "expected_output_tokens": spec.expected_output_tokens,
                    "max_output_tokens": spec.max_output_tokens,
                }
                for spec in self._specs
            ]
        }


def load_catalog(path: Path | None = None) -> ArmCatalog:
    """Load and validate ``config/pricing.toml`` (or another pricing file)."""
    pricing_path = path if path is not None else DEFAULT_PRICING_PATH
    try:
        raw: Mapping[str, object] = tomllib.loads(pricing_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArmConfigError(f"pricing config {pricing_path} not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ArmConfigError(
            f"pricing config {pricing_path.name} is not valid TOML: {exc}"
        ) from exc
    try:
        parsed = _PricingFile.model_validate(raw)
    except ValueError as exc:
        raise ArmConfigError(f"pricing config {pricing_path.name} is invalid: {exc}") from exc
    return ArmCatalog(parsed.arm)
