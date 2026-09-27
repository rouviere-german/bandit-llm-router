"""Seeded synthetic workload: prompt families with a latent difficulty.

Five families stand in for the mix a router actually sees -- ``extraction``, ``summarization``,
``reasoning``, ``code`` and ``small_talk``. Each request carries a latent difficulty in [0, 1] drawn
from a family-specific Beta distribution. Difficulty is *not* an input to the router: it shapes the
prompt (length, structure, digit density, fenced code) and the models' quality, so the featurizer
can only see its traces. That is the point of the simulator -- a policy that beats static rules has
to infer difficulty from the same signals a production router would have.

Everything is derived from one integer seed. :attr:`Workload.digest` hashes the seed and the family
configuration, and that digest is written into every decision log row and MLflow run, so a result
can be traced back to the exact workload that produced it.

Requests come from a fixed pool (default 4096) that is sampled with replacement. Real traffic
repeats prompts too, and a pool lets the featurizer's cache do its job, which is what keeps a
200k-request simulation to seconds of CPU instead of minutes of string hashing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from bandit_llm_router.core.canonical import canonical_json, sha256_hex
from bandit_llm_router.features import MAX_PROMPT_CHARS, RoutingContext

__all__ = [
    "FAMILIES",
    "FAMILY_SPECS",
    "WORKLOAD_VERSION",
    "FamilySpec",
    "SyntheticRequest",
    "Workload",
]

WORKLOAD_VERSION: Final = "wl-v1"
IntArray = npt.NDArray[np.int64]

_WORDS: Final[tuple[str, ...]] = (
    "invoice",
    "shipment",
    "latency",
    "customer",
    "cluster",
    "rollout",
    "policy",
    "budget",
    "quarter",
    "revenue",
    "outage",
    "contract",
    "warehouse",
    "forecast",
    "supplier",
    "refund",
    "ticket",
    "backlog",
    "release",
    "incident",
    "threshold",
    "retention",
    "pipeline",
    "schema",
    "migration",
    "dashboard",
    "anomaly",
    "capacity",
    "discount",
    "renewal",
    "escalation",
    "inventory",
    "settlement",
    "compliance",
    "throughput",
    "allocation",
    "handover",
    "variance",
)
_FIELDS: Final[tuple[str, ...]] = (
    "order_id",
    "amount_eur",
    "due_date",
    "account",
    "region",
    "sku",
    "quantity",
    "tax_rate",
    "carrier",
    "priority",
)
_IDENTIFIERS: Final[tuple[str, ...]] = (
    "total",
    "rows",
    "items",
    "window",
    "budget",
    "score",
    "limit",
    "batch",
)


class FamilySpec(BaseModel):
    """Configuration of one prompt family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z_]{0,31}$")
    weight: float = Field(gt=0.0)
    style: str = Field(pattern=r"^(fields|prose|steps|code|chat)$")
    difficulty_alpha: float = Field(gt=0.0)
    difficulty_beta: float = Field(gt=0.0)
    min_units: int = Field(ge=1, le=200)
    max_units: int = Field(ge=1, le=200)
    base_output_tokens: int = Field(ge=32, le=2_048)

    def digest_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


FAMILY_SPECS: Final[tuple[FamilySpec, ...]] = (
    FamilySpec(
        name="extraction",
        weight=0.30,
        style="fields",
        difficulty_alpha=2.0,
        difficulty_beta=4.0,
        min_units=3,
        max_units=16,
        base_output_tokens=160,
    ),
    FamilySpec(
        name="summarization",
        weight=0.22,
        style="prose",
        difficulty_alpha=2.5,
        difficulty_beta=2.5,
        min_units=4,
        max_units=26,
        base_output_tokens=260,
    ),
    FamilySpec(
        name="reasoning",
        weight=0.12,
        style="steps",
        difficulty_alpha=4.0,
        difficulty_beta=2.0,
        min_units=3,
        max_units=14,
        base_output_tokens=420,
    ),
    FamilySpec(
        name="code",
        weight=0.14,
        style="code",
        difficulty_alpha=3.0,
        difficulty_beta=2.2,
        min_units=4,
        max_units=20,
        base_output_tokens=380,
    ),
    FamilySpec(
        name="small_talk",
        weight=0.22,
        style="chat",
        difficulty_alpha=1.4,
        difficulty_beta=6.0,
        min_units=1,
        max_units=3,
        base_output_tokens=80,
    ),
)
FAMILIES: Final[tuple[str, ...]] = tuple(spec.name for spec in FAMILY_SPECS)


@dataclass(frozen=True, slots=True)
class SyntheticRequest:
    """One pooled request: its validated context, its family and its latent difficulty."""

    context: RoutingContext
    family: str
    difficulty: float


def _words(rng: np.random.Generator, count: int) -> list[str]:
    return [_WORDS[index] for index in rng.integers(0, len(_WORDS), size=count)]


def _fields_body(rng: np.random.Generator, units: int) -> str:
    lines = []
    for _ in range(units):
        field = _FIELDS[int(rng.integers(0, len(_FIELDS)))]
        value = int(rng.integers(100, 999_999))
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def _prose_body(rng: np.random.Generator, units: int) -> str:
    sentences = []
    for _ in range(units):
        sentences.append(" ".join(_words(rng, int(rng.integers(8, 18)))).capitalize() + ".")
    return " ".join(sentences)


def _steps_body(rng: np.random.Generator, units: int) -> str:
    lines = []
    for index in range(units):
        budget = int(rng.integers(2, 90))
        lines.append(
            f"{index + 1}. Constraint {' '.join(_words(rng, 4))} must stay under {budget}."
        )
    return "\n".join(lines)


def _code_body(rng: np.random.Generator, units: int) -> str:
    lines = ["```python"]
    for index in range(units):
        name = _IDENTIFIERS[int(rng.integers(0, len(_IDENTIFIERS)))]
        lines.append(f"    {name}_{index} = compute({int(rng.integers(1, 64))}, {name})")
    lines.append("```")
    return "\n".join(lines)


def _chat_body(rng: np.random.Generator, units: int) -> str:
    return " ".join(" ".join(_words(rng, 5)) for _ in range(units))


_HEADERS: Final[dict[str, tuple[str, ...]]] = {
    "fields": (
        "Extract every field as JSON:",
        "Pull the structured values out of this record:",
    ),
    "prose": (
        "Summarize the following notes in three bullets:",
        "Write a short digest of these notes:",
    ),
    "steps": (
        "Work through the constraints and give the schedule that satisfies all of them:",
        "Decide which plan satisfies every constraint below, and say why:",
    ),
    "code": (
        "Review this snippet and rewrite it correctly:",
        "Find the bug in the following code and fix it:",
    ),
    "chat": ("Hi, quick question:", "Hey there --"),
}
_TRAILERS: Final[dict[str, str]] = {
    "fields": "Return only the fields you found.",
    "prose": "Keep it under 80 words.",
    "steps": "Show the reasoning that rules out the alternatives.",
    "code": "Explain what was wrong?",
    "chat": "Thanks!",
}


def _build_prompt(rng: np.random.Generator, spec: FamilySpec, difficulty: float) -> str:
    span = spec.max_units - spec.min_units
    units = round(spec.min_units + difficulty * span)
    units = max(spec.min_units, min(spec.max_units, units))
    header = _HEADERS[spec.style][int(rng.integers(0, len(_HEADERS[spec.style])))]
    if spec.style == "fields":
        body = _fields_body(rng, units)
    elif spec.style == "prose":
        body = _prose_body(rng, units)
    elif spec.style == "steps":
        body = _steps_body(rng, units)
    elif spec.style == "code":
        body = _code_body(rng, units)
    else:
        body = _chat_body(rng, units)
    prompt = f"{header}\n{body}\n{_TRAILERS[spec.style]}"
    return prompt[:MAX_PROMPT_CHARS]


class Workload:
    """A seeded pool of synthetic requests plus the digest that identifies it."""

    __slots__ = ("_digest", "_pool", "_seed", "_specs", "_tenants", "_weights")

    def __init__(
        self,
        *,
        seed: int,
        pool_size: int = 4_096,
        tenants: Sequence[str] = ("tenant_a", "tenant_b"),
        specs: Sequence[FamilySpec] = FAMILY_SPECS,
    ) -> None:
        if pool_size < len(specs):
            raise ValueError("pool_size must be at least one request per family")
        if not tenants:
            raise ValueError("at least one tenant is required")
        self._seed = seed
        self._specs = tuple(specs)
        self._tenants = tuple(tenants)
        weights = np.asarray([spec.weight for spec in self._specs], dtype=np.float64)
        self._weights = weights / weights.sum()
        self._digest = sha256_hex(
            canonical_json(
                {
                    "workload_version": WORKLOAD_VERSION,
                    "seed": seed,
                    "pool_size": pool_size,
                    "tenants": list(self._tenants),
                    "families": [spec.digest_payload() for spec in self._specs],
                }
            )
        )
        self._pool = self._build_pool(pool_size)

    def _build_pool(self, pool_size: int) -> tuple[SyntheticRequest, ...]:
        rng = np.random.default_rng(self._seed)
        family_indices = rng.choice(len(self._specs), size=pool_size, p=self._weights)
        requests: list[SyntheticRequest] = []
        for position in range(pool_size):
            spec = self._specs[int(family_indices[position])]
            difficulty = float(rng.beta(spec.difficulty_alpha, spec.difficulty_beta))
            prompt = _build_prompt(rng, spec, difficulty)
            output_tokens = int(
                min(1_024, max(64, round(spec.base_output_tokens * (0.6 + 0.9 * difficulty))))
            )
            tenant = self._tenants[int(rng.integers(0, len(self._tenants)))]
            tier = "priority" if rng.random() < 0.15 else "standard"
            requests.append(
                SyntheticRequest(
                    context=RoutingContext(
                        tenant_id=tenant,
                        prompt=prompt,
                        max_output_tokens=output_tokens,
                        tier=tier,
                    ),
                    family=spec.name,
                    difficulty=difficulty,
                )
            )
        return tuple(requests)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def digest(self) -> str:
        """Workload hash: seed plus family configuration. Logged with every decision."""
        return self._digest

    @property
    def pool(self) -> tuple[SyntheticRequest, ...]:
        return self._pool

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    def family_indices(self) -> IntArray:
        """Family index of every pooled request (array-friendly for vectorized quality models)."""
        lookup = {spec.name: index for index, spec in enumerate(self._specs)}
        return np.asarray([lookup[request.family] for request in self._pool], dtype=np.int64)

    def difficulties(self) -> npt.NDArray[np.float64]:
        return np.asarray([request.difficulty for request in self._pool], dtype=np.float64)

    def sample_indices(self, count: int, rng: np.random.Generator) -> IntArray:
        """Draw ``count`` pool indices with replacement (the request stream)."""
        if count < 0:
            raise ValueError("count must be non-negative")
        return rng.integers(0, len(self._pool), size=count, dtype=np.int64)
