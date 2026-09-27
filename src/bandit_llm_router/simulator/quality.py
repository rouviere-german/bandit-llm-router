"""Per-arm quality response model, with scheduled shifts.

Expected quality of arm ``k`` on a request of family ``f`` with latent difficulty ``d`` is

    q(k, f, d) = sigmoid(intercept[k, f] - slope[k] * d)

Bigger models have a higher intercept *and* a flatter slope: they degrade more slowly as a request
gets harder. That single fact is what makes routing a contextual problem -- on easy requests the
arms are nearly tied and cost decides, on hard reasoning and code the gap is worth paying for. The
numbers below are chosen to make that trade-off non-trivial, not measured from real models;
MODEL_CARD.md says so plainly.

Observed quality adds mean-zero Gaussian noise clipped to [0, 1], so a single observation is a weak
signal and the learner has to average.

A :class:`QualityShift` degrades one (family, arm) intercept from a given step onward -- a model
update that quietly gets worse at one thing. The environment turns a shift schedule into a small
number of *regimes* and precomputes one expected-quality matrix per regime, which is what keeps a
200k-step simulation fast and makes discounted-versus-stationary posteriors comparable on exactly
the same stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from bandit_llm_router.simulator.workload import FAMILIES

__all__ = [
    "DEFAULT_ARM_QUALITY",
    "ArmQualitySpec",
    "QualityModel",
    "QualityShift",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class ArmQualitySpec(BaseModel):
    """Quality curve of one arm: a per-family intercept and one difficulty slope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    intercepts: Mapping[str, float]
    difficulty_slope: float = Field(gt=0.0, le=10.0)
    noise_sd: float = Field(default=0.08, ge=0.0, le=0.5)

    def digest_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "intercepts": {name: self.intercepts[name] for name in sorted(self.intercepts)},
            "difficulty_slope": self.difficulty_slope,
            "noise_sd": self.noise_sd,
        }


# Intercepts are ``strength[arm] + ease[family]`` with
#   ease   = {small_talk: 2.2, extraction: 1.2, summarization: 0.9, code: 0.2, reasoning: -0.2}
#   strength/slope = {haiku: 2.3 / 4.6, sonnet: 1.3 / 2.2, opus: 1.7 / 0.9}
# so every arm is near-saturated on trivial requests and they diverge as difficulty rises -- which
# is exactly where the routing decision lives. The slopes are what separate the arms: Haiku falls
# off five times faster than Opus. These numbers are a design choice that makes the cost-quality
# trade-off non-trivial (each arm is optimal on part of the traffic), not a measurement of the real
# models; see MODEL_CARD.md.
DEFAULT_ARM_QUALITY: tuple[ArmQualitySpec, ...] = (
    ArmQualitySpec(
        arm="haiku",
        intercepts={
            "extraction": 3.5,
            "summarization": 3.2,
            "reasoning": 2.1,
            "code": 2.5,
            "small_talk": 4.5,
        },
        difficulty_slope=4.6,
    ),
    ArmQualitySpec(
        arm="sonnet",
        intercepts={
            "extraction": 2.5,
            "summarization": 2.2,
            "reasoning": 1.1,
            "code": 1.5,
            "small_talk": 3.5,
        },
        difficulty_slope=2.2,
    ),
    ArmQualitySpec(
        arm="opus",
        intercepts={
            "extraction": 2.9,
            "summarization": 2.6,
            "reasoning": 1.5,
            "code": 1.9,
            "small_talk": 3.9,
        },
        difficulty_slope=0.9,
    ),
)


@dataclass(frozen=True, slots=True)
class QualityShift:
    """From ``at_step`` on, ``arm``'s intercept for ``family`` moves by ``intercept_delta``."""

    at_step: int
    family: str
    arm: str
    intercept_delta: float

    def digest_payload(self) -> dict[str, object]:
        return {
            "at_step": self.at_step,
            "family": self.family,
            "arm": self.arm,
            "intercept_delta": self.intercept_delta,
        }


def _sigmoid(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-values))


class QualityModel:
    """Expected and observed quality for a set of arms over a set of families."""

    __slots__ = ("_arms", "_families", "_intercepts", "_noise_sd", "_slopes", "_specs")

    def __init__(
        self,
        specs: Sequence[ArmQualitySpec] = DEFAULT_ARM_QUALITY,
        *,
        families: Sequence[str] = FAMILIES,
    ) -> None:
        if not specs:
            raise ValueError("at least one arm quality spec is required")
        self._specs = tuple(specs)
        self._arms = tuple(spec.arm for spec in specs)
        self._families = tuple(families)
        missing = [
            (spec.arm, family)
            for spec in specs
            for family in self._families
            if family not in spec.intercepts
        ]
        if missing:
            raise ValueError(f"missing intercepts for {missing}")
        self._intercepts = np.asarray(
            [[spec.intercepts[family] for family in self._families] for spec in specs],
            dtype=np.float64,
        )
        self._slopes = np.asarray([spec.difficulty_slope for spec in specs], dtype=np.float64)
        self._noise_sd = np.asarray([spec.noise_sd for spec in specs], dtype=np.float64)

    @property
    def arms(self) -> tuple[str, ...]:
        return self._arms

    @property
    def families(self) -> tuple[str, ...]:
        return self._families

    @property
    def noise_sd(self) -> FloatArray:
        return self._noise_sd

    def digest_payload(self) -> dict[str, object]:
        return {
            "families": list(self._families),
            "arms": [spec.digest_payload() for spec in self._specs],
        }

    def shifted(self, shifts: Sequence[QualityShift]) -> QualityModel:
        """A copy with every shift's intercept delta applied (regime construction)."""
        specs: list[ArmQualitySpec] = []
        for spec in self._specs:
            intercepts = dict(spec.intercepts)
            for shift in shifts:
                if shift.arm != spec.arm:
                    continue
                if shift.family not in intercepts:
                    raise ValueError(f"shift names unknown family {shift.family!r}")
                intercepts[shift.family] += shift.intercept_delta
            specs.append(spec.model_copy(update={"intercepts": intercepts}))
        return QualityModel(specs, families=self._families)

    def expected(self, family: str, arm: str, difficulty: float) -> float:
        """Expected quality in (0, 1) for one (family, arm, difficulty) triple."""
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        arm_index = self._arms.index(arm)
        family_index = self._families.index(family)
        logit = self._intercepts[arm_index, family_index] - self._slopes[arm_index] * difficulty
        return float(_sigmoid(np.asarray(logit, dtype=np.float64)))

    def expected_matrix(self, family_indices: IntArray, difficulties: FloatArray) -> FloatArray:
        """Expected quality for every (request, arm) pair: shape ``(n, len(arms))``."""
        families = np.asarray(family_indices, dtype=np.int64)
        difficulty = np.asarray(difficulties, dtype=np.float64)
        if families.shape != difficulty.shape:
            raise ValueError("family_indices and difficulties must have the same shape")
        if families.size and (families.min() < 0 or families.max() >= len(self._families)):
            raise ValueError("family index out of range")
        if np.any(difficulty < 0.0) or np.any(difficulty > 1.0):
            raise ValueError("difficulty must be in [0, 1]")
        logits = self._intercepts[:, families].T - np.outer(difficulty, self._slopes)
        return _sigmoid(logits)

    def sample(self, expected: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Observed quality: expected plus per-arm Gaussian noise, clipped to [0, 1]."""
        values = np.asarray(expected, dtype=np.float64)
        noise = rng.normal(0.0, 1.0, size=values.shape)
        scale = self._noise_sd if values.ndim == 1 else self._noise_sd[np.newaxis, :]
        return np.clip(values + noise * scale, 0.0, 1.0)
