"""Non-learning baselines: fixed arms and a hand-written routing rule.

These are the incumbents a learned policy has to beat, and they are also the honest reference points
for the cost claims: "30% cheaper" means nothing without saying cheaper *than what*. Three shapes:

* :class:`StaticArm` -- always route to one arm (always-Haiku, always-Sonnet, always-Opus).
* :class:`KeywordRule` -- the rule a team writes first: code goes to the big model, long prompts to
  the mid model, everything else to the cheap one. It reads the same features the bandits see, so
  the comparison is about the decision rule and not about the inputs.

Both are usable as logging policies too, which is why they accept a propensity floor: a fully
deterministic logging policy produces logs on which no other policy can be evaluated, because every
unchosen arm has propensity zero.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from bandit_llm_router.features import DENSE_FEATURE_NAMES, HASH_BUCKETS
from bandit_llm_router.policies.base import (
    PolicySnapshot,
    apply_propensity_floor,
    policy_version,
)

__all__ = ["KeywordRule", "StaticArm", "StaticPolicy", "feature_index"]

FloatArray = npt.NDArray[np.float64]
_LENGTH_SCALE: Final = 8.0


def feature_index(name: str) -> int:
    """Column of a named dense feature in a feature vector."""
    return HASH_BUCKETS + DENSE_FEATURE_NAMES.index(name)


class StaticPolicy:
    """A policy that never learns. Subclasses only decide which arm wins a context."""

    name = "static"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        propensity_floor: float = 0.0,
        feature_version: str = "",
    ) -> None:
        if len(arms) < 2:
            raise ValueError("at least two arms are required")
        if propensity_floor < 0.0 or propensity_floor * len(arms) > 1.0:
            raise ValueError("propensity_floor * n_arms must be in [0, 1]")
        self.arms = arms
        self.n_arms = len(arms)
        self.dim = dim
        self.propensity_floor = propensity_floor
        self.feature_version = feature_version

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {}

    def params_payload(self) -> dict[str, JsonValue]:
        return {
            "arms": list(self.arms),
            "dim": self.dim,
            "propensity_floor": self.propensity_floor,
            "feature_version": self.feature_version,
            **self.hyperparameters(),
        }

    @property
    def version(self) -> str:
        return policy_version(self.name, self.params_payload())

    @property
    def updates(self) -> int:
        return 0

    def chosen_arms(self, features: FloatArray) -> npt.NDArray[np.int64]:
        """Arm index chosen for each row of ``features``."""
        raise NotImplementedError

    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        del rng
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise ValueError(f"features must have shape ({self.dim},)")
        chosen = int(self.chosen_arms(vector[np.newaxis, :])[0])
        base = np.zeros(self.n_arms, dtype=np.float64)
        base[chosen] = 1.0
        return apply_propensity_floor(base, self.propensity_floor)

    def probabilities_matrix(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        del rng
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"features must have shape (n, {self.dim})")
        chosen = self.chosen_arms(matrix)
        base = np.zeros((matrix.shape[0], self.n_arms), dtype=np.float64)
        base[np.arange(matrix.shape[0]), chosen] = 1.0
        floor = self.propensity_floor
        return (1.0 - floor * self.n_arms) * base + floor

    def update(
        self, features: FloatArray, arm_index: int, reward: float, weight: float = 1.0
    ) -> None:
        """No-op: static policies do not learn."""
        del features, arm_index, reward, weight

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            name=self.name,
            version=self.version,
            params=dict(self.params_payload()),
            arrays={"static": np.zeros(1, dtype=np.float64)},
        )


class StaticArm(StaticPolicy):
    """Always route to ``target``."""

    name = "static_arm"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        target: str,
        propensity_floor: float = 0.0,
        feature_version: str = "",
    ) -> None:
        super().__init__(
            arms=arms, dim=dim, propensity_floor=propensity_floor, feature_version=feature_version
        )
        if target not in arms:
            raise ValueError(f"target {target!r} is not one of the arms {arms}")
        self.target = target
        self.target_index = arms.index(target)
        self.name = f"static_{target}"

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {"target": self.target}

    def chosen_arms(self, features: FloatArray) -> npt.NDArray[np.int64]:
        return np.full(features.shape[0], self.target_index, dtype=np.int64)


class KeywordRule(StaticPolicy):
    """The rule a team writes before measuring anything: code -> big, long -> mid, else cheap."""

    name = "keyword_rule"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        code_arm: str,
        long_arm: str,
        default_arm: str,
        long_prompt_chars: int = 900,
        propensity_floor: float = 0.0,
        feature_version: str = "",
    ) -> None:
        super().__init__(
            arms=arms, dim=dim, propensity_floor=propensity_floor, feature_version=feature_version
        )
        for arm in (code_arm, long_arm, default_arm):
            if arm not in arms:
                raise ValueError(f"arm {arm!r} is not one of {arms}")
        if long_prompt_chars < 1:
            raise ValueError("long_prompt_chars must be >= 1")
        self.code_arm = code_arm
        self.long_arm = long_arm
        self.default_arm = default_arm
        self.long_prompt_chars = long_prompt_chars
        self._code_index = arms.index(code_arm)
        self._long_index = arms.index(long_arm)
        self._default_index = arms.index(default_arm)
        self._code_feature = feature_index("has_code_fence")
        self._length_feature = feature_index("log_chars")
        self._length_threshold = float(np.log1p(long_prompt_chars) / _LENGTH_SCALE)

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {
            "code_arm": self.code_arm,
            "long_arm": self.long_arm,
            "default_arm": self.default_arm,
            "long_prompt_chars": self.long_prompt_chars,
        }

    def chosen_arms(self, features: FloatArray) -> npt.NDArray[np.int64]:
        has_code = features[:, self._code_feature] > 0.5
        is_long = features[:, self._length_feature] >= self._length_threshold
        chosen = np.full(features.shape[0], self._default_index, dtype=np.int64)
        chosen[is_long] = self._long_index
        chosen[has_code] = self._code_index
        return chosen
