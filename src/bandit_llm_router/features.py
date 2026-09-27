"""Context featurizer and the data contract for a routing decision's input.

A routing decision sees one request. The policy needs a fixed-width numeric context for it, and
the log must be useful for off-policy evaluation months later, while the prompt itself is never
persisted. This module is where those three requirements meet:

* :class:`RoutingContext` is the validated request contract (bounded prompt length, bounded output
  request, an arm-agnostic tenant id). Nothing downstream accepts raw strings.
* :class:`Featurizer` turns a context into a fixed vector: a hashing vectorizer over character
  n-grams (scikit-learn's :class:`~sklearn.feature_extraction.text.HashingVectorizer` with
  ``analyzer="char_wb"``, so nothing is downloaded and no vocabulary is fitted or stored) plus
  hand-built length, structure and metadata features and a bias term.
* Every vector is checked against :func:`validate_features`: fixed dimension, finite, and bounded
  by :data:`FEATURE_ABS_MAX`. A contract violation raises instead of reaching a policy, because an
  unbounded feature would silently destabilize the ridge posteriors and the importance weights.

The prompt is used to compute features and is then dropped. What is logged next to the features is
:attr:`FeatureResult.context_digest`, a keyed BLAKE2b digest of the prompt: it lets logs dedupe and
join without holding text. The key comes from settings; with an empty key the digest of a short
prompt is guessable by anyone who can enumerate candidates (documented in docs/THREAT_MODEL.md),
so deployments that care set ``APP_FEATURE_DIGEST_KEY``.

The featurizer is deterministic: the same context yields the same vector in any process, on any
platform. That is what makes a logged feature row replayable, so :attr:`Featurizer.version` is part
of every policy version and every log row.
"""

from __future__ import annotations

import hashlib
import string
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bandit_llm_router.core.canonical import canonical_json, sha256_hex

__all__ = [
    "DENSE_FEATURE_NAMES",
    "FEATURE_ABS_MAX",
    "FEATURE_DIM",
    "FEATURE_VERSION",
    "HASH_BUCKETS",
    "MAX_PROMPT_CHARS",
    "FeatureContractError",
    "FeatureResult",
    "Featurizer",
    "RoutingContext",
]

FloatArray = npt.NDArray[np.float64]

FEATURE_VERSION: Final = "ctx-v1"
HASH_BUCKETS: Final = 24
NGRAM_RANGE: Final = (3, 5)
MAX_PROMPT_CHARS: Final = 8_000
FEATURE_ABS_MAX: Final = 4.0
_DIGEST_BYTES: Final = 16
_LENGTH_SCALE: Final = 8.0
_WORD_LENGTH_SCALE: Final = 12.0
_NEWLINE_SCALE: Final = 40.0
_PUNCTUATION = frozenset(string.punctuation)

DENSE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_chars",
    "digit_ratio",
    "punctuation_ratio",
    "uppercase_ratio",
    "newline_density",
    "whitespace_ratio",
    "has_code_fence",
    "has_question",
    "avg_word_length",
    "distinct_word_ratio",
    "log_max_output_tokens",
    "is_priority_tier",
)
FEATURE_DIM: Final = HASH_BUCKETS + len(DENSE_FEATURE_NAMES) + 1

TenantId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")]


class FeatureContractError(ValueError):
    """A feature vector violated its contract (shape, finiteness or magnitude)."""


class RoutingContext(BaseModel):
    """One routing request. ``prompt`` is read to build features and is never persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: TenantId
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_output_tokens: int = Field(default=512, ge=1, le=8_192)
    tier: Literal["standard", "priority"] = "standard"

    @property
    def estimated_input_tokens(self) -> int:
        """Deterministic 4-characters-per-token estimate, used for cost *estimates*."""
        return max(1, -(-len(self.prompt) // 4))

    @property
    def input_token_ceiling(self) -> int:
        """Conservative 2-characters-per-token bound, used for budget *reservations*.

        Real tokenizers average near four characters per token on English prose but can approach one
        on dense code or non-Latin scripts. Reservations use the tighter bound so the amount
        committed after a call is almost never larger than the amount held before it; an overshoot
        is logged and, because the guard never charges more than it reserved, capped at one call.
        """
        return max(1, -(-len(self.prompt) // 2))


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """A context's feature vector plus the keyed digest that identifies it in logs."""

    vector: FloatArray
    context_digest: str
    estimated_input_tokens: int


def validate_features(vector: FloatArray, *, dim: int = FEATURE_DIM) -> FloatArray:
    """Return ``vector`` if it satisfies the feature contract, else raise."""
    if vector.ndim != 1 or vector.shape[0] != dim:
        raise FeatureContractError(f"expected a vector of shape ({dim},), got {vector.shape}")
    if vector.dtype != np.float64:
        raise FeatureContractError(f"expected float64 features, got {vector.dtype}")
    if not np.all(np.isfinite(vector)):
        raise FeatureContractError("feature vector contains NaN or infinite values")
    if float(np.max(np.abs(vector))) > FEATURE_ABS_MAX:
        raise FeatureContractError(f"feature vector exceeds the |x| <= {FEATURE_ABS_MAX} bound")
    return vector


def validate_feature_matrix(matrix: FloatArray, *, dim: int = FEATURE_DIM) -> FloatArray:
    """Contract check for a batch of feature rows (same rules as :func:`validate_features`)."""
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise FeatureContractError(f"expected a matrix with {dim} columns, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise FeatureContractError("feature matrix contains NaN or infinite values")
    if matrix.size and float(np.max(np.abs(matrix))) > FEATURE_ABS_MAX:
        raise FeatureContractError(f"feature matrix exceeds the |x| <= {FEATURE_ABS_MAX} bound")
    return matrix


def _dense_features(context: RoutingContext) -> FloatArray:
    prompt = context.prompt
    length = len(prompt)
    digits = sum(char.isdigit() for char in prompt)
    punctuation = sum(char in _PUNCTUATION for char in prompt)
    uppercase = sum(char.isupper() for char in prompt)
    whitespace = sum(char.isspace() for char in prompt)
    newlines = prompt.count("\n")
    words = prompt.split()
    word_count = len(words)
    avg_word_length = (sum(len(word) for word in words) / word_count) if word_count else 0.0
    distinct_ratio = (len({word.lower() for word in words}) / word_count) if word_count else 0.0
    return np.asarray(
        [
            np.log1p(length) / _LENGTH_SCALE,
            digits / length,
            punctuation / length,
            uppercase / length,
            min(newlines / _NEWLINE_SCALE, 1.0),
            whitespace / length,
            1.0 if "```" in prompt else 0.0,
            1.0 if "?" in prompt else 0.0,
            min(avg_word_length / _WORD_LENGTH_SCALE, 1.0),
            distinct_ratio,
            np.log1p(context.max_output_tokens) / _LENGTH_SCALE,
            1.0 if context.tier == "priority" else 0.0,
        ],
        dtype=np.float64,
    )


class Featurizer:
    """Deterministic context -> fixed-width feature vector, with an LRU cache.

    The cache is keyed by the context digest, so repeated prompts (which dominate any real
    workload, and every simulated one) are featurized once. Cached entries hold vectors and
    digests, never prompt text.
    """

    def __init__(
        self,
        *,
        buckets: int = HASH_BUCKETS,
        ngram_range: tuple[int, int] = NGRAM_RANGE,
        digest_key: bytes = b"",
        cache_size: int = 8_192,
    ) -> None:
        if buckets < 2:
            raise ValueError("buckets must be >= 2")
        if not 1 <= ngram_range[0] <= ngram_range[1]:
            raise ValueError("ngram_range must satisfy 1 <= low <= high")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        if len(digest_key) > hashlib.blake2b.MAX_KEY_SIZE:
            raise ValueError("digest_key is too long for BLAKE2b")
        from sklearn.feature_extraction.text import HashingVectorizer  # noqa: PLC0415

        self._buckets = buckets
        self._ngram_range = ngram_range
        self._digest_key = bytes(digest_key)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[FloatArray, int]] = OrderedDict()
        self._vectorizer: Any = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=buckets,
            alternate_sign=True,
            norm="l2",
            lowercase=True,
            dtype=np.float64,
        )
        self.dim = buckets + len(DENSE_FEATURE_NAMES) + 1
        self.version = f"{FEATURE_VERSION}-{self.spec_digest[:12]}"

    @property
    def spec_digest(self) -> str:
        """Hash of the featurizer's configuration (not of the digest key)."""
        return sha256_hex(
            canonical_json(
                {
                    "feature_version": FEATURE_VERSION,
                    "buckets": self._buckets,
                    "ngram_range": list(self._ngram_range),
                    "dense": list(DENSE_FEATURE_NAMES),
                    "abs_max": FEATURE_ABS_MAX,
                }
            )
        )

    def feature_names(self) -> tuple[str, ...]:
        return (
            *(f"hash_{index}" for index in range(self._buckets)),
            *DENSE_FEATURE_NAMES,
            "bias",
        )

    def digest(self, prompt: str) -> str:
        """Keyed BLAKE2b digest of a prompt, the only prompt-derived value that is stored."""
        hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES, key=self._digest_key)
        hasher.update(FEATURE_VERSION.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(prompt.encode("utf-8"))
        return hasher.hexdigest()

    def _hashed_block(self, prompts: Sequence[str]) -> FloatArray:
        matrix = self._vectorizer.transform(list(prompts))
        return np.asarray(matrix.toarray(), dtype=np.float64)

    def transform(self, context: RoutingContext) -> FeatureResult:
        """Featurize one context, using and updating the LRU cache."""
        digest = self.digest(context.prompt)
        key = f"{digest}:{context.max_output_tokens}:{context.tier}"
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return FeatureResult(cached[0], digest, cached[1])
        vector = np.concatenate(
            [
                self._hashed_block([context.prompt])[0],
                _dense_features(context),
                np.ones(1, dtype=np.float64),
            ]
        )
        validate_features(vector, dim=self.dim)
        vector.setflags(write=False)
        tokens = context.estimated_input_tokens
        self._cache[key] = (vector, tokens)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return FeatureResult(vector, digest, tokens)

    def transform_batch(self, contexts: Sequence[RoutingContext]) -> FloatArray:
        """Featurize many contexts at once (one vectorizer call), bypassing the cache."""
        if not contexts:
            return np.zeros((0, self.dim), dtype=np.float64)
        hashed = self._hashed_block([context.prompt for context in contexts])
        dense = np.stack([_dense_features(context) for context in contexts])
        bias = np.ones((len(contexts), 1), dtype=np.float64)
        matrix = np.concatenate([hashed, dense, bias], axis=1)
        return validate_feature_matrix(matrix, dim=self.dim)
