"""Off-policy evaluation: IPS, SNIPS and doubly robust, with bootstrap CIs and ESS.

Given a log of ``(context, action, propensity, reward)`` rows produced by whatever policy was live,
these estimators answer "what would a *different* policy have earned?". All three are standard, and
they are all here because they fail in different ways -- which is the point of reporting them
together:

* **IPS** ``(1/n) sum w_i r_i`` with ``w_i = pi(a_i | x_i) / p_i`` is unbiased when the log has
  positive propensity everywhere the candidate acts, and has terrible variance when the candidate
  disagrees with the logging policy. Optional weight clipping trades that variance for bias.
* **SNIPS** divides by ``sum w_i`` instead of ``n``. It is biased but usually far more stable, and
it
  cannot report a value outside the reward range, which IPS can.
* **Doubly robust** (Dudik, Langford and Li, *Doubly Robust Policy Evaluation and Learning*, ICML
  2011) adds a reward model ``qhat``: ``sum_a pi(a|x) qhat(x,a)`` plus an importance-weighted
  correction of the model's residual. It is consistent if *either* the propensities or the reward
  model are right, and its variance is much lower than IPS when ``qhat`` is decent.

The reward model is a per-arm ridge regression fitted by K-fold **cross-fitting**: each row's
``qhat`` comes from a model that never saw that row (Chernozhukov et al., *Double/Debiased Machine
Learning*, 2018). Fitting in-sample makes the residual correction look smaller than it is and
quietly biases DR toward the reward model; cross-fitting costs one extra fit per fold and removes
that failure mode.

Uncertainty is reported two ways: an analytic standard error (a ratio-estimator standard error for
SNIPS) and a percentile bootstrap over rows. The bootstrap holds ``qhat`` fixed, so the DR interval
covers sampling noise in the log but not the reward model's own uncertainty; the coverage that
matters is therefore measured, not assumed -- ``bench/bench_ope.py`` reports empirical 95% coverage
against the simulator's exact policy values.

:func:`effective_sample_size` is the honest health check on every one of these numbers:
``(sum w)^2 / sum w^2``. When the candidate concentrates on actions the log rarely took, ESS
collapses and the estimate is a few rows wearing a confidence interval. Policy screening refuses to
promote anything whose ESS is too small.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from bandit_llm_router.core.canonical import canonical_json, sha256_hex

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "CrossFitRidgeRewardModel",
    "Estimate",
    "LoggedBatch",
    "OPEReport",
    "doubly_robust",
    "effective_sample_size",
    "evaluate_policy",
    "importance_weights",
    "ips",
    "snips",
    "stack_probabilities",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_BOOTSTRAP: Final = 400
_MIN_PROPENSITY: Final = 1e-12


@dataclass(frozen=True, slots=True)
class LoggedBatch:
    """A validated block of logged decisions with their rewards.

    Rows without a reward have no place here: they are dropped when the batch is built, because an
    importance-weighted estimator has nothing to weight without an outcome.
    """

    features: FloatArray
    actions: IntArray
    propensities: FloatArray
    rewards: FloatArray
    n_arms: int

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a 2-D array")
        rows = self.features.shape[0]
        for name, array in (
            ("actions", self.actions),
            ("propensities", self.propensities),
            ("rewards", self.rewards),
        ):
            if array.shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},), got {array.shape}")
        if rows == 0:
            raise ValueError("a logged batch must contain at least one row")
        if self.n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if int(self.actions.min()) < 0 or int(self.actions.max()) >= self.n_arms:
            raise ValueError("actions contain an arm index outside the arm set")
        if np.any(self.propensities <= _MIN_PROPENSITY):
            raise ValueError("propensities must be positive: a zero-propensity row is unusable")
        if np.any(self.propensities > 1.0 + 1e-9):
            raise ValueError("propensities cannot exceed 1")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("features contain NaN or infinite values")
        if not np.all(np.isfinite(self.rewards)):
            raise ValueError("rewards contain NaN or infinite values")

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def digest(self) -> str:
        """Content hash of the batch, so a screening decision is bound to the data it used."""
        parts = [
            np.ascontiguousarray(self.features, dtype=np.float64).tobytes(),
            np.ascontiguousarray(self.actions, dtype=np.int64).tobytes(),
            np.ascontiguousarray(self.propensities, dtype=np.float64).tobytes(),
            np.ascontiguousarray(self.rewards, dtype=np.float64).tobytes(),
        ]
        return sha256_hex(b"".join(parts) + canonical_json({"n_arms": self.n_arms}).encode())

    def chosen_probabilities(self, target_probabilities: FloatArray) -> FloatArray:
        """``pi(a_i | x_i)`` for the logged actions."""
        matrix = _check_target(target_probabilities, self.n, self.n_arms)
        return matrix[np.arange(self.n), self.actions]


def _check_target(target_probabilities: FloatArray, rows: int, n_arms: int) -> FloatArray:
    matrix = np.asarray(target_probabilities, dtype=np.float64)
    if matrix.shape != (rows, n_arms):
        raise ValueError(f"target probabilities must have shape ({rows}, {n_arms})")
    if np.any(matrix < -1e-9):
        raise ValueError("target probabilities cannot be negative")
    totals = matrix.sum(axis=1)
    if np.any(np.abs(totals - 1.0) > 1e-6):
        raise ValueError("every row of target probabilities must sum to 1")
    return matrix


@dataclass(frozen=True, slots=True)
class Estimate:
    """One estimate of a policy's per-request value."""

    method: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    ess: float
    clipped_fraction: float = 0.0

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "method": self.method,
            "value": self.value,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "ess": self.ess,
            "clipped_fraction": self.clipped_fraction,
        }


def importance_weights(
    batch: LoggedBatch, target_probabilities: FloatArray, *, clip: float | None = None
) -> tuple[FloatArray, float]:
    """``pi(a|x) / p`` per row, optionally clipped. Returns the weights and the clipped fraction."""
    chosen = batch.chosen_probabilities(target_probabilities)
    weights = chosen / batch.propensities
    if clip is None:
        return weights, 0.0
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    clipped = np.minimum(weights, clip)
    fraction = float(np.mean(weights > clip))
    return clipped, fraction


def effective_sample_size(weights: FloatArray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``; 0 when every weight is 0."""
    array = np.asarray(weights, dtype=np.float64)
    total = float(array.sum())
    square = float(np.square(array).sum())
    return 0.0 if square <= 0.0 else total * total / square


def _percentile_ci(samples: FloatArray, alpha: float) -> tuple[float, float]:
    low, high = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _bootstrap_means(values: FloatArray, bootstrap: int, rng: np.random.Generator) -> FloatArray:
    rows = values.shape[0]
    out = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        picks = rng.integers(0, rows, size=rows)
        out[index] = float(values[picks].mean())
    return out


def _bootstrap_ratios(
    numerator: FloatArray, denominator: FloatArray, bootstrap: int, rng: np.random.Generator
) -> FloatArray:
    rows = numerator.shape[0]
    out = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        picks = rng.integers(0, rows, size=rows)
        total = float(denominator[picks].sum())
        out[index] = float(numerator[picks].sum()) / total if total > 0.0 else 0.0
    return out


def _estimate_from_values(
    method: str,
    values: FloatArray,
    weights: FloatArray,
    *,
    bootstrap: int,
    alpha: float,
    rng: np.random.Generator | None,
    clipped_fraction: float,
) -> Estimate:
    rows = values.shape[0]
    value = float(values.mean())
    std_error = float(values.std(ddof=1) / np.sqrt(rows)) if rows > 1 else 0.0
    if bootstrap > 0:
        generator = rng if rng is not None else np.random.default_rng(0)
        ci_low, ci_high = _percentile_ci(_bootstrap_means(values, bootstrap, generator), alpha)
    else:
        half = 1.959963984540054 * std_error
        ci_low, ci_high = value - half, value + half
    return Estimate(
        method=method,
        value=value,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n=rows,
        ess=effective_sample_size(weights),
        clipped_fraction=clipped_fraction,
    )


def ips(
    batch: LoggedBatch,
    target_probabilities: FloatArray,
    *,
    clip: float | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Estimate:
    """Inverse propensity scoring."""
    weights, clipped = importance_weights(batch, target_probabilities, clip=clip)
    return _estimate_from_values(
        "ips",
        weights * batch.rewards,
        weights,
        bootstrap=bootstrap,
        alpha=alpha,
        rng=rng,
        clipped_fraction=clipped,
    )


def snips(
    batch: LoggedBatch,
    target_probabilities: FloatArray,
    *,
    clip: float | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Estimate:
    """Self-normalized IPS: the same weights, divided by their own sum."""
    weights, clipped = importance_weights(batch, target_probabilities, clip=clip)
    numerator = weights * batch.rewards
    total_weight = float(weights.sum())
    value = float(numerator.sum() / total_weight) if total_weight > 0.0 else 0.0
    rows = batch.n
    if rows > 1 and total_weight > 0.0:
        residual = weights * (batch.rewards - value)
        std_error = float(residual.std(ddof=1) / (np.sqrt(rows) * weights.mean()))
    else:
        std_error = 0.0
    if bootstrap > 0:
        generator = rng if rng is not None else np.random.default_rng(0)
        ci_low, ci_high = _percentile_ci(
            _bootstrap_ratios(numerator, weights, bootstrap, generator), alpha
        )
    else:
        half = 1.959963984540054 * std_error
        ci_low, ci_high = value - half, value + half
    return Estimate(
        method="snips",
        value=value,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n=rows,
        ess=effective_sample_size(weights),
        clipped_fraction=clipped,
    )


class CrossFitRidgeRewardModel:
    """Per-arm ridge reward model, fitted by K-fold cross-fitting.

    ``fit_predict`` returns ``qhat(x_i, a)`` for every row and arm, where row ``i``'s predictions
    come from the folds that exclude ``i``. Arms with no out-of-fold observations fall back to the
    out-of-fold mean reward, which keeps DR defined on logs where an arm is almost never taken.
    """

    def __init__(self, *, ridge: float = 1.0, folds: int = 4) -> None:
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if folds < 2:
            raise ValueError("folds must be >= 2 for cross-fitting")
        self.ridge = ridge
        self.folds = folds

    def _fit_arm(self, features: FloatArray, rewards: FloatArray) -> FloatArray:
        gram = features.T @ features + self.ridge * np.eye(features.shape[1])
        return np.linalg.solve(gram, features.T @ rewards)

    def fit_predict(self, batch: LoggedBatch, rng: np.random.Generator | None = None) -> FloatArray:
        generator = rng if rng is not None else np.random.default_rng(0)
        rows = batch.n
        folds = min(self.folds, rows)
        assignment = generator.permutation(rows) % folds
        predictions = np.zeros((rows, batch.n_arms), dtype=np.float64)
        for fold in range(folds):
            held_out = assignment == fold
            train = ~held_out
            if not np.any(held_out):  # pragma: no cover - only with folds > rows
                continue
            if not np.any(train):
                predictions[held_out, :] = float(batch.rewards.mean())
                continue
            fallback = float(batch.rewards[train].mean())
            for arm in range(batch.n_arms):
                selector = train & (batch.actions == arm)
                if not np.any(selector):
                    predictions[held_out, arm] = fallback
                    continue
                theta = self._fit_arm(batch.features[selector], batch.rewards[selector])
                predictions[held_out, arm] = batch.features[held_out] @ theta
        return predictions


def doubly_robust(
    batch: LoggedBatch,
    target_probabilities: FloatArray,
    q_hat: FloatArray,
    *,
    clip: float | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Estimate:
    """Doubly robust estimate given cross-fitted reward-model predictions ``q_hat``."""
    matrix = _check_target(target_probabilities, batch.n, batch.n_arms)
    model = np.asarray(q_hat, dtype=np.float64)
    if model.shape != matrix.shape:
        raise ValueError("q_hat must have the same shape as the target probabilities")
    if not np.all(np.isfinite(model)):
        raise ValueError("q_hat contains NaN or infinite values")
    weights, clipped = importance_weights(batch, matrix, clip=clip)
    direct = np.einsum("ij,ij->i", matrix, model)
    residual = batch.rewards - model[np.arange(batch.n), batch.actions]
    values = direct + weights * residual
    return _estimate_from_values(
        "dr",
        values,
        weights,
        bootstrap=bootstrap,
        alpha=alpha,
        rng=rng,
        clipped_fraction=clipped,
    )


@dataclass(frozen=True, slots=True)
class OPEReport:
    """IPS, SNIPS and DR for one candidate on one log, plus the log's health numbers."""

    ips: Estimate
    snips: Estimate
    dr: Estimate
    n: int
    ess: float
    min_propensity: float
    mean_logged_reward: float

    def as_dict(self) -> dict[str, object]:
        return {
            "ips": self.ips.as_dict(),
            "snips": self.snips.as_dict(),
            "dr": self.dr.as_dict(),
            "n": self.n,
            "ess": self.ess,
            "min_propensity": self.min_propensity,
            "mean_logged_reward": self.mean_logged_reward,
        }


def evaluate_policy(
    batch: LoggedBatch,
    target_probabilities: FloatArray,
    *,
    ridge: float = 1.0,
    folds: int = 4,
    clip: float | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    q_hat: FloatArray | None = None,
) -> OPEReport:
    """Run all three estimators on one candidate's action probabilities."""
    generator = rng if rng is not None else np.random.default_rng(0)
    matrix = _check_target(target_probabilities, batch.n, batch.n_arms)
    model = (
        q_hat
        if q_hat is not None
        else CrossFitRidgeRewardModel(ridge=ridge, folds=folds).fit_predict(batch, generator)
    )
    ips_estimate = ips(
        batch, matrix, clip=clip, bootstrap=bootstrap, alpha=alpha, rng=np.random.default_rng(1)
    )
    snips_estimate = snips(
        batch, matrix, clip=clip, bootstrap=bootstrap, alpha=alpha, rng=np.random.default_rng(2)
    )
    dr_estimate = doubly_robust(
        batch,
        matrix,
        model,
        clip=clip,
        bootstrap=bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(3),
    )
    return OPEReport(
        ips=ips_estimate,
        snips=snips_estimate,
        dr=dr_estimate,
        n=batch.n,
        ess=dr_estimate.ess,
        min_propensity=float(batch.propensities.min()),
        mean_logged_reward=float(batch.rewards.mean()),
    )


def stack_probabilities(rows: Sequence[FloatArray]) -> FloatArray:
    """Stack per-row distributions into the ``(n, n_arms)`` matrix the estimators expect."""
    if not rows:
        raise ValueError("at least one row is required")
    return np.vstack([np.asarray(row, dtype=np.float64) for row in rows])
