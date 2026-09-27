"""The three learning policies: LinUCB, linear Thompson sampling and linear epsilon-greedy.

All three share :class:`~bandit_llm_router.policies.linear.DiscountedRidge` posteriors and differ
only in how they turn per-arm means and uncertainties into a distribution over arms:

* **LinUCB** scores arm ``k`` as ``theta_k . x + alpha * sqrt(x^T A_k^-1 x)`` and puts its mass on
  the best score (Li et al., WWW 2010).
* **Thompson sampling** draws one reward sample per arm from the posterior and picks the best draw
  (Thompson 1933; Chapelle and Li, *An Empirical Evaluation of Thompson Sampling*, NeurIPS 2011).
  For a single context the posterior of ``theta_k . x`` is Gaussian with mean ``theta_k . x`` and
  standard deviation ``sigma * sqrt(x^T A_k^-1 x)``, and the arms' posteriors are independent, so a
  draw costs one standard normal per arm rather than a ``d``-dimensional sample.
* **Epsilon-greedy** ignores uncertainty and mixes the greedy arm with uniform exploration.

Each then applies the propensity floor, so the logged propensity of any action is at least
``propensity_floor``.

**Why Thompson sampling logs Monte-Carlo probabilities.** Sampling an arm from a posterior draw is a
randomized decision whose action probabilities have no closed form, and logging the realized argmax
with probability 1 would make every later off-policy estimate wrong. So the deployed policy *is* the
Monte-Carlo distribution: ``mc_samples`` posterior draws are taken, the fraction of draws won by
each arm (after the floor) is the distribution, and the action is sampled from it. The logged
propensity is then exactly the probability with which that action was taken, conditional on the
draws -- which is all importance weighting needs, because the draws are independent of the reward.
ADR 0003 records the alternatives that were rejected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from bandit_llm_router.policies.base import (
    DEFAULT_PROPENSITY_FLOOR,
    PolicySnapshot,
    apply_propensity_floor,
    greedy_probabilities,
    policy_version,
)
from bandit_llm_router.policies.linear import DiscountedRidge

__all__ = [
    "EpsilonGreedy",
    "LinUCB",
    "LinearBanditPolicy",
    "LinearThompson",
    "ScoreBasedPolicy",
]

FloatArray = npt.NDArray[np.float64]
_SNAPSHOT_VERSION: Final = 1


class LinearBanditPolicy(ABC):
    """Shared plumbing: posteriors, versioning, online updates and snapshots."""

    name: str = "linear"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        ridge: float = 1.0,
        discount: float = 1.0,
        propensity_floor: float = DEFAULT_PROPENSITY_FLOOR,
        feature_version: str = "",
        refresh_every: int = 2_048,
    ) -> None:
        if len(arms) < 2:
            raise ValueError("at least two arms are required")
        if len(set(arms)) != len(arms):
            raise ValueError("arm names must be unique")
        if propensity_floor < 0.0 or propensity_floor * len(arms) > 1.0:
            raise ValueError("propensity_floor * n_arms must be in [0, 1]")
        self.arms = arms
        self.n_arms = len(arms)
        self.dim = dim
        self.propensity_floor = propensity_floor
        self.feature_version = feature_version
        self.posterior = DiscountedRidge(
            self.n_arms, dim, ridge=ridge, discount=discount, refresh_every=refresh_every
        )

    # --- identity -------------------------------------------------------------------------

    @abstractmethod
    def hyperparameters(self) -> dict[str, JsonValue]:
        """Policy-specific hyperparameters (part of the version)."""

    def params_payload(self) -> dict[str, JsonValue]:
        return {
            "arms": list(self.arms),
            "dim": self.dim,
            "ridge": self.posterior.ridge,
            "discount": self.posterior.discount,
            "propensity_floor": self.propensity_floor,
            "feature_version": self.feature_version,
            **self.hyperparameters(),
        }

    @property
    def version(self) -> str:
        return policy_version(self.name, self.params_payload())

    @property
    def updates(self) -> int:
        return self.posterior.updates

    # --- decisions ------------------------------------------------------------------------

    @abstractmethod
    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Action distribution for one context."""

    @abstractmethod
    def probabilities_matrix(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Action distributions for a batch of contexts, shape ``(n, n_arms)``."""

    def _check_matrix(self, features: FloatArray) -> FloatArray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"features must have shape (n, {self.dim})")
        return matrix

    # --- learning -------------------------------------------------------------------------

    def update(
        self, features: FloatArray, arm_index: int, reward: float, weight: float = 1.0
    ) -> None:
        """Absorb one (context, action, reward) observation."""
        self.posterior.update(arm_index, features, reward, weight)

    # --- persistence ----------------------------------------------------------------------

    def snapshot(self) -> PolicySnapshot:
        """Everything needed to rebuild this policy: JSON params plus float arrays."""
        return PolicySnapshot(
            name=self.name,
            version=self.version,
            params={
                "snapshot_version": _SNAPSHOT_VERSION,
                "policy_name": self.name,
                "policy_version": self.version,
                "updates": self.posterior.updates,
                **self.params_payload(),
            },
            arrays=self.posterior.arrays(),
        )

    def load_snapshot(self, snapshot: PolicySnapshot) -> None:
        """Restore posteriors from a snapshot of a policy with the same configuration version."""
        if snapshot.version != self.version:
            raise ValueError(
                f"snapshot version {snapshot.version} does not match policy version {self.version}"
            )
        self.posterior = DiscountedRidge.from_arrays(
            dict(snapshot.arrays),
            ridge=self.posterior.ridge,
            discount=self.posterior.discount,
            refresh_every=self.posterior.refresh_every,
        )


class ScoreBasedPolicy(LinearBanditPolicy):
    """A policy that ranks arms by a deterministic score and then applies the floor."""

    @abstractmethod
    def scores(self, features: FloatArray) -> FloatArray:
        """Per-arm scores for one context (higher is better)."""

    @abstractmethod
    def scores_matrix(self, features: FloatArray) -> FloatArray:
        """Per-arm scores for a batch, shape ``(n, n_arms)``."""

    def _mix(self, greedy: FloatArray) -> FloatArray:
        """Hook for subclasses that explore beyond the floor (epsilon-greedy)."""
        return greedy

    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        del rng  # deterministic given the posteriors
        greedy = greedy_probabilities(self.scores(features))
        return apply_propensity_floor(self._mix(greedy), self.propensity_floor)

    def probabilities_matrix(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        del rng
        scores = self.scores_matrix(self._check_matrix(features))
        best = scores == scores.max(axis=1, keepdims=True)
        greedy = best / best.sum(axis=1, keepdims=True)
        mixed = self._mix(greedy)
        floor = self.propensity_floor
        return (1.0 - floor * self.n_arms) * mixed + floor


class LinUCB(ScoreBasedPolicy):
    """Disjoint LinUCB with an optional discount and a propensity floor."""

    name = "linucb"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        ridge: float = 1.0,
        discount: float = 1.0,
        propensity_floor: float = DEFAULT_PROPENSITY_FLOOR,
        feature_version: str = "",
        refresh_every: int = 2_048,
        alpha: float = 0.6,
    ) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        super().__init__(
            arms=arms,
            dim=dim,
            ridge=ridge,
            discount=discount,
            propensity_floor=propensity_floor,
            feature_version=feature_version,
            refresh_every=refresh_every,
        )

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {"alpha": self.alpha}

    def scores(self, features: FloatArray) -> FloatArray:
        return self.posterior.means(features) + self.alpha * self.posterior.widths(features)

    def scores_matrix(self, features: FloatArray) -> FloatArray:
        return self.posterior.means_matrix(features) + self.alpha * self.posterior.widths_matrix(
            features
        )


class EpsilonGreedy(ScoreBasedPolicy):
    """Linear epsilon-greedy: the ridge point estimate plus uniform exploration."""

    name = "epsilon_greedy"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        ridge: float = 1.0,
        discount: float = 1.0,
        propensity_floor: float = DEFAULT_PROPENSITY_FLOOR,
        feature_version: str = "",
        refresh_every: int = 2_048,
        epsilon: float = 0.1,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        super().__init__(
            arms=arms,
            dim=dim,
            ridge=ridge,
            discount=discount,
            propensity_floor=propensity_floor,
            feature_version=feature_version,
            refresh_every=refresh_every,
        )

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {"epsilon": self.epsilon}

    def scores(self, features: FloatArray) -> FloatArray:
        return self.posterior.means(features)

    def scores_matrix(self, features: FloatArray) -> FloatArray:
        return self.posterior.means_matrix(features)

    def _mix(self, greedy: FloatArray) -> FloatArray:
        return (1.0 - self.epsilon) * greedy + self.epsilon / self.n_arms


class LinearThompson(LinearBanditPolicy):
    """Thompson sampling over the ridge posteriors, logging Monte-Carlo action probabilities."""

    name = "thompson"

    def __init__(
        self,
        *,
        arms: tuple[str, ...],
        dim: int,
        ridge: float = 1.0,
        discount: float = 1.0,
        propensity_floor: float = DEFAULT_PROPENSITY_FLOOR,
        feature_version: str = "",
        refresh_every: int = 2_048,
        posterior_sd: float = 0.25,
        mc_samples: int = 24,
    ) -> None:
        if posterior_sd <= 0.0:
            raise ValueError("posterior_sd must be positive")
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        self.posterior_sd = posterior_sd
        self.mc_samples = mc_samples
        super().__init__(
            arms=arms,
            dim=dim,
            ridge=ridge,
            discount=discount,
            propensity_floor=propensity_floor,
            feature_version=feature_version,
            refresh_every=refresh_every,
        )

    def hyperparameters(self) -> dict[str, JsonValue]:
        return {"posterior_sd": self.posterior_sd, "mc_samples": self.mc_samples}

    def probabilities(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        means = self.posterior.means(features)
        widths = self.posterior.widths(features)
        draws = means + self.posterior_sd * widths * rng.normal(size=(self.mc_samples, self.n_arms))
        counts = np.bincount(np.argmax(draws, axis=1), minlength=self.n_arms).astype(np.float64)
        return apply_propensity_floor(counts / self.mc_samples, self.propensity_floor)

    def probabilities_matrix(self, features: FloatArray, rng: np.random.Generator) -> FloatArray:
        matrix = self._check_matrix(features)
        means = self.posterior.means_matrix(matrix)
        widths = self.posterior.widths_matrix(matrix)
        counts = np.zeros_like(means)
        rows = np.arange(matrix.shape[0])
        for _ in range(self.mc_samples):
            draws = means + self.posterior_sd * widths * rng.normal(size=means.shape)
            counts[rows, np.argmax(draws, axis=1)] += 1.0
        floor = self.propensity_floor
        return (1.0 - floor * self.n_arms) * (counts / self.mc_samples) + floor
