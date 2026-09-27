"""The simulated routing environment: one array-shaped view of workload, quality and cost.

Everything a simulation needs is precomputed once per *regime* (a stretch of steps between quality
shifts) and then read by index:

* ``feature_matrix`` -- the pool's features, computed once by the featurizer.
* ``cost_matrix`` / ``ceiling_matrix`` -- expected and worst-case cost of every (request, arm) pair
  in micro-USD, from the pricing config.
* per regime, ``expected_quality`` (the noise-free quality curve, which the sampler adds noise to)
  and ``expected_reward`` (the true mean of the reward a call actually produces).

A step is then an array lookup plus one Gaussian draw, which is why 200k requests take seconds. It
also makes the oracle exact: because the expected reward of every arm is known, cumulative regret is
measured against the best *expected* arm rather than against a noisy realized best, so the regret
curve reflects the policy's decisions and not the sampling noise.

**Why ``expected_reward`` is integrated rather than shaped from the expected quality.** A realized
call draws ``quality = clip(q(x, a) + noise, 0, 1)`` and earns ``clip(quality - penalty, 0, 1)``.
Both clips are censoring, so the reward of the *expected* quality is not the expected reward: on
this workload roughly seven in ten (request, arm) pairs sit within three standard deviations of
the upper clip, and shaping the mean directly overstates the truth by up to 0.03 per cell. That gap
is small enough to look like noise and large enough to matter twice over -- it would move the
oracle's choice of best arm, and it would show up as a systematic bias in *every* off-policy
estimator, including IPS, whose whole claim is that it has none. So the expectation is integrated
over the noise with a fixed Gaussian quadrature (:data:`_QUADRATURE_NODES` normal quantile
midpoints, deterministic and seed-free), which makes ``expected_reward`` exactly the mean of
what the sampler produces.

The environment never touches a provider. Live routing goes through the provider layer in
``bandit_llm_router.service``; this module only produces the rewards a simulated day would produce.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bandit_llm_router.arms import ArmCatalog
from bandit_llm_router.core.canonical import canonical_json, sha256_hex
from bandit_llm_router.features import Featurizer
from bandit_llm_router.reward import RewardConfig
from bandit_llm_router.simulator.quality import QualityModel, QualityShift
from bandit_llm_router.simulator.workload import Workload

__all__ = ["Outcome", "Regime", "SimulatedEnvironment"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Nodes for the noise integral. 256 midpoint quantiles put the quadrature error around 1e-5, two
# orders of magnitude below the censoring effect being integrated.
_QUADRATURE_NODES = 256


@functools.cache
def _quadrature_nodes() -> FloatArray:
    """Standard-normal quantile midpoints: a deterministic stand-in for averaging over draws."""
    from scipy import stats  # noqa: PLC0415 - scipy is slow to import; only pay for it here

    midpoints = (np.arange(_QUADRATURE_NODES, dtype=np.float64) + 0.5) / _QUADRATURE_NODES
    nodes: FloatArray = np.asarray(stats.norm.ppf(midpoints), dtype=np.float64)
    nodes.setflags(write=False)
    return nodes


@dataclass(frozen=True, slots=True)
class Regime:
    """Expected quality and reward for the pool during one stretch of steps.

    Two references, because they answer different questions:

    * ``best_arm`` / ``best_reward`` -- the unconstrained oracle, which no policy in this repo can
      reach: it needs no exploration and is not restricted to any model class.
    * ``linear_best_arm`` / ``linear_best_reward`` -- the clairvoyant member of the *policy class*:
      the greedy policy of the best least-squares linear fit of the true expected rewards on the
      features. This is the right reference for regret, because a linear bandit on a nonlinear
      world has an irreducible approximation gap, and charging a learner for that gap makes its
      regret grow linearly however well it learns.
    """

    start_step: int
    expected_quality: FloatArray
    expected_reward: FloatArray
    best_arm: IntArray
    best_reward: FloatArray
    linear_best_arm: IntArray
    linear_best_reward: FloatArray


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a simulated call returned."""

    arm_index: int
    quality: float
    cost_micros: float
    reward: float


class SimulatedEnvironment:
    """Workload + quality model + pricing, precomputed per regime."""

    __slots__ = (
        "_catalog",
        "_ceiling_matrix",
        "_cost_matrix",
        "_feature_matrix",
        "_featurizer",
        "_input_tokens",
        "_quality",
        "_regimes",
        "_reward_config",
        "_shifts",
        "_workload",
    )

    def __init__(
        self,
        *,
        workload: Workload,
        catalog: ArmCatalog,
        quality: QualityModel | None = None,
        reward_config: RewardConfig | None = None,
        shifts: Sequence[QualityShift] = (),
        featurizer: Featurizer | None = None,
    ) -> None:
        self._workload = workload
        self._catalog = catalog
        self._quality = quality if quality is not None else QualityModel()
        self._reward_config = reward_config if reward_config is not None else RewardConfig()
        if self._quality.arms != catalog.names:
            raise ValueError(
                f"quality model arms {self._quality.arms} do not match catalog {catalog.names}"
            )
        self._shifts = tuple(sorted(shifts, key=lambda shift: shift.at_step))
        if any(shift.at_step < 0 for shift in self._shifts):
            raise ValueError("quality shifts cannot start before step 0")
        self._featurizer = featurizer if featurizer is not None else Featurizer()
        contexts = [request.context for request in workload.pool]
        self._feature_matrix = self._featurizer.transform_batch(contexts)
        self._input_tokens = np.asarray(
            [context.estimated_input_tokens for context in contexts], dtype=np.int64
        )
        self._cost_matrix, self._ceiling_matrix = self._build_cost_matrices()
        self._regimes = self._build_regimes()

    # --- construction helpers -------------------------------------------------------------

    def _build_cost_matrices(self) -> tuple[FloatArray, FloatArray]:
        requested_out = np.asarray(
            [request.context.max_output_tokens for request in self._workload.pool], dtype=np.int64
        )
        return self._catalog.cost_matrices(self._input_tokens, requested_out)

    def _integrate_expected_reward(
        self, expected_quality: FloatArray, model: QualityModel
    ) -> FloatArray:
        """True ``E[reward | x, a]``, integrated over the quality noise.

        The integrand is the realized reward as the sampler computes it, so the two cannot drift
        apart. Gaussian quadrature over normal quantile midpoints is used rather than sampling:
        it is deterministic, needs no seed, and its error at this node count is far below the
        censoring effect it exists to capture (see the module docstring).
        """
        penalty = self._reward_config.cost_weight * np.minimum(
            1.0, self._cost_matrix / float(self._reward_config.cost_reference_micros)
        )
        nodes = _quadrature_nodes()
        noise_sd = model.noise_sd[np.newaxis, :, np.newaxis]
        quality = np.clip(
            expected_quality[:, :, np.newaxis] + nodes[np.newaxis, np.newaxis, :] * noise_sd,
            0.0,
            1.0,
        )
        rewards = np.clip(quality - penalty[:, :, np.newaxis], 0.0, 1.0)
        return np.asarray(rewards.mean(axis=2), dtype=np.float64)

    def _best_linear_policy(self, expected_reward: FloatArray) -> tuple[IntArray, FloatArray]:
        """The clairvoyant policy inside the learner's own model class.

        Per arm, the least-squares linear fit of the true expected reward on the features is the
        best any linear model could do; its greedy policy is then the best a linear bandit could
        converge to. The value returned is that policy's *true* expected reward, not its own
        prediction, so the reference is achievable rather than optimistic.

        Without this, regret is measured against a target the model class cannot represent: the
        learner converges, the residual approximation gap keeps accruing at a constant rate, and the
        fitted log-log slope creeps toward 1 -- which reads as "stopped learning" when in fact it
        means "learned everything a linear model can".
        """
        features = self._feature_matrix
        thetas, *_ = np.linalg.lstsq(features, expected_reward, rcond=None)
        predicted = features @ thetas
        chosen = np.argmax(predicted, axis=1).astype(np.int64)
        value = expected_reward[np.arange(expected_reward.shape[0]), chosen]
        return chosen, np.asarray(value, dtype=np.float64)

    def _build_regimes(self) -> tuple[Regime, ...]:
        family_indices = self._workload.family_indices()
        difficulties = self._workload.difficulties()
        boundaries = [0, *sorted({shift.at_step for shift in self._shifts})]
        regimes: list[Regime] = []
        for start in boundaries:
            active = [shift for shift in self._shifts if shift.at_step <= start]
            model = self._quality.shifted(active) if active else self._quality
            expected_quality = model.expected_matrix(family_indices, difficulties)
            expected_reward = self._integrate_expected_reward(expected_quality, model)
            best_arm = np.argmax(expected_reward, axis=1).astype(np.int64)
            best_reward = expected_reward[np.arange(expected_reward.shape[0]), best_arm]
            linear_best_arm, linear_best_reward = self._best_linear_policy(expected_reward)
            regimes.append(
                Regime(
                    start_step=start,
                    linear_best_arm=linear_best_arm,
                    linear_best_reward=linear_best_reward,
                    expected_quality=expected_quality,
                    expected_reward=expected_reward,
                    best_arm=best_arm,
                    best_reward=best_reward,
                )
            )
        return tuple(regimes)

    # --- shape and identity ---------------------------------------------------------------

    @property
    def workload(self) -> Workload:
        return self._workload

    @property
    def catalog(self) -> ArmCatalog:
        return self._catalog

    @property
    def featurizer(self) -> Featurizer:
        return self._featurizer

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_config

    @property
    def arms(self) -> tuple[str, ...]:
        return self._catalog.names

    @property
    def n_arms(self) -> int:
        return len(self._catalog)

    @property
    def pool_size(self) -> int:
        return len(self._workload.pool)

    @property
    def feature_dim(self) -> int:
        return int(self._feature_matrix.shape[1])

    @property
    def feature_matrix(self) -> FloatArray:
        return self._feature_matrix

    @property
    def cost_matrix(self) -> FloatArray:
        """Expected cost in micro-USD of every (request, arm) pair."""
        return self._cost_matrix

    @property
    def ceiling_matrix(self) -> FloatArray:
        """Worst-case cost in micro-USD of every (request, arm) pair."""
        return self._ceiling_matrix

    @property
    def input_tokens(self) -> IntArray:
        return self._input_tokens

    @property
    def noise_sd(self) -> FloatArray:
        """Per-arm quality noise standard deviation (the hot loop scales pre-drawn normals)."""
        return self._quality.noise_sd

    @property
    def regimes(self) -> tuple[Regime, ...]:
        return self._regimes

    @property
    def shifts(self) -> tuple[QualityShift, ...]:
        return self._shifts

    def digest_payload(self) -> dict[str, object]:
        return {
            "workload": self._workload.digest,
            "quality": self._quality.digest_payload(),
            "reward": self._reward_config.digest_payload(),
            "arms": self._catalog.digest_payload(),
            "shifts": [shift.digest_payload() for shift in self._shifts],
            "featurizer": self._featurizer.version,
        }

    @property
    def digest(self) -> str:
        """Hash of the whole environment: workload, quality curves, prices, shifts, features."""
        return sha256_hex(canonical_json(self.digest_payload()))

    # --- per-step access ------------------------------------------------------------------

    def regime_at(self, step: int) -> Regime:
        """The regime governing ``step`` (regimes are ordered and start at 0)."""
        if step < 0:
            raise ValueError("step must be non-negative")
        chosen = self._regimes[0]
        for regime in self._regimes:
            if regime.start_step <= step:
                chosen = regime
            else:
                break
        return chosen

    def expected_rewards(self, index: int, step: int) -> FloatArray:
        row: FloatArray = self.regime_at(step).expected_reward[index]
        return row

    def best_arm(self, index: int, step: int) -> int:
        return int(self.regime_at(step).best_arm[index])

    def draw_noise(self, count: int, rng: np.random.Generator) -> FloatArray:
        """Pre-draw standard normals so the hot loop does no RNG work per step."""
        return rng.normal(0.0, 1.0, size=count)

    def outcome(self, index: int, arm_index: int, step: int, standard_normal: float) -> Outcome:
        """Realize one call: noisy quality, its cost and the shaped reward."""
        regime = self.regime_at(step)
        sd = float(self._quality.noise_sd[arm_index])
        quality = float(
            np.clip(regime.expected_quality[index, arm_index] + standard_normal * sd, 0.0, 1.0)
        )
        cost = float(self._cost_matrix[index, arm_index])
        penalty = self._reward_config.cost_weight * min(
            1.0, cost / self._reward_config.cost_reference_micros
        )
        reward = float(min(1.0, max(0.0, quality - penalty)))
        return Outcome(arm_index=arm_index, quality=quality, cost_micros=cost, reward=reward)
