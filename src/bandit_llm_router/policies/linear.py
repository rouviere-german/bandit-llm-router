"""Per-arm discounted ridge posteriors, shared by LinUCB, Thompson sampling and epsilon-greedy.

Each arm keeps Gram statistics over the contexts it was chosen on:

    A_k  <-  discount * A_k + (1 - discount) * ridge * I + w * x x^T
    b_k  <-  discount * b_k + w * r * x
    theta_k = A_k^-1 b_k

With ``discount = 1`` this is textbook ridge regression per arm (Li et al., *A Contextual-Bandit
Approach to Personalized News Article Recommendation*, WWW 2010). With ``discount < 1`` old
observations decay exponentially, which is how the router survives a model update that changes an
arm's quality: the posterior forgets the pre-shift world at a controlled rate, and the
re-regularization term ``(1 - discount) * ridge * I`` keeps ``A_k`` from collapsing toward zero in
directions that stop being explored. The effect on the confidence width is the point -- uncertainty
grows again where evidence has aged, so exploration restarts on its own. This is a deliberately
simple variant of the weighted linear bandit of Russac, Vernade and Cappe (*Weighted Linear Bandits
for Non-Stationary Environments*, NeurIPS 2019), which tracks a second matrix to tighten the
confidence bound; ADR 0004 records why the simpler form is enough here.

One inverse matters and it is cached per arm:

* ``A_k^-1`` for the mean and the LinUCB width. When ``discount == 1`` an update is a rank-one
  change, so the inverse is maintained by the Sherman-Morrison identity in O(d^2) instead of
  O(d^3). Rank-one updates accumulate floating-point error, so every ``refresh_every`` updates the
  inverse is recomputed exactly; ``tests/unit/test_policies.py`` checks the maintained inverse
  against a fresh solve.
Thompson sampling needs no Cholesky factor here: for a single context the posterior of
``theta_k . x`` is ``N(theta_k . x, sigma^2 x^T A_k^-1 x)``, so a draw is the mean plus
``sigma * width * z`` with one standard normal per arm. See ``policies/linear_bandit.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["DiscountedRidge"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class DiscountedRidge:
    """Ridge posteriors for ``n_arms`` arms over ``dim``-dimensional contexts."""

    __slots__ = (
        "_gram",
        "_gram_inverse",
        "_inverse_fresh",
        "_mass",
        "_rhs",
        "_since_refresh",
        "_theta",
        "_theta_fresh",
        "_updates",
        "dim",
        "discount",
        "n_arms",
        "refresh_every",
        "ridge",
    )

    def __init__(
        self,
        n_arms: int,
        dim: int,
        *,
        ridge: float = 1.0,
        discount: float = 1.0,
        refresh_every: int = 2_048,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        self.n_arms = n_arms
        self.dim = dim
        self.ridge = ridge
        self.discount = discount
        self.refresh_every = refresh_every
        eye = np.eye(dim, dtype=np.float64)
        self._gram = np.repeat((ridge * eye)[np.newaxis, :, :], n_arms, axis=0)
        self._gram_inverse = np.repeat((eye / ridge)[np.newaxis, :, :], n_arms, axis=0)
        self._rhs = np.zeros((n_arms, dim), dtype=np.float64)
        self._theta = np.zeros((n_arms, dim), dtype=np.float64)
        self._inverse_fresh = np.ones(n_arms, dtype=np.bool_)
        self._theta_fresh = np.ones(n_arms, dtype=np.bool_)
        self._mass = np.zeros(n_arms, dtype=np.float64)
        self._since_refresh = np.zeros(n_arms, dtype=np.int64)
        self._updates = 0

    # --- state ----------------------------------------------------------------------------

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def mass(self) -> FloatArray:
        """Discounted observation mass per arm (an effective sample size for the posterior)."""
        return self._mass.copy()

    def _ensure_inverse(self, arm: int) -> FloatArray:
        if not self._inverse_fresh[arm]:
            self._gram_inverse[arm] = np.linalg.inv(self._gram[arm])
            self._inverse_fresh[arm] = True
            self._since_refresh[arm] = 0
        inverse: FloatArray = self._gram_inverse[arm]
        return inverse

    def _ensure_theta(self, arm: int) -> FloatArray:
        if not self._theta_fresh[arm]:
            self._theta[arm] = self._ensure_inverse(arm) @ self._rhs[arm]
            self._theta_fresh[arm] = True
        theta: FloatArray = self._theta[arm]
        return theta

    def inverse(self, arm: int) -> FloatArray:
        """``A_k^-1`` (cached)."""
        return self._ensure_inverse(arm)

    def theta(self, arm: int) -> FloatArray:
        """Ridge point estimate for one arm (cached)."""
        return self._ensure_theta(arm)

    def theta_matrix(self) -> FloatArray:
        """All arms' point estimates, shape ``(n_arms, dim)``."""
        for arm in range(self.n_arms):
            self._ensure_theta(arm)
        return self._theta

    # --- updates --------------------------------------------------------------------------

    def update(self, arm: int, features: FloatArray, reward: float, weight: float = 1.0) -> None:
        """Absorb one observation for ``arm``. ``weight`` scales its influence (trust weighting)."""
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm index {arm} out of range")
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise ValueError(f"features must have shape ({self.dim},), got {vector.shape}")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if self.discount < 1.0:
            decay = self.discount
            self._gram[arm] *= decay
            self._gram[arm] += (1.0 - decay) * self.ridge * np.eye(self.dim)
            self._gram[arm] += weight * np.outer(vector, vector)
            self._rhs[arm] = decay * self._rhs[arm] + weight * reward * vector
            self._mass[arm] = decay * self._mass[arm] + weight
            self._inverse_fresh[arm] = False
        else:
            self._gram[arm] += weight * np.outer(vector, vector)
            self._rhs[arm] += weight * reward * vector
            self._mass[arm] += weight
            self._since_refresh[arm] += 1
            if self._since_refresh[arm] >= self.refresh_every:
                self._inverse_fresh[arm] = False
            elif weight > 0.0:
                inverse = self._gram_inverse[arm]
                projected = inverse @ vector
                denominator = 1.0 + weight * float(vector @ projected)
                if denominator <= 0.0:  # pragma: no cover - guards against a non-PSD update
                    self._inverse_fresh[arm] = False
                else:
                    inverse -= (weight / denominator) * np.outer(projected, projected)
        self._theta_fresh[arm] = False
        self._updates += 1

    # --- predictions ----------------------------------------------------------------------

    def means(self, features: FloatArray) -> FloatArray:
        """Predicted reward of every arm for one context, shape ``(n_arms,)``."""
        return self.theta_matrix() @ np.asarray(features, dtype=np.float64)

    def widths(self, features: FloatArray) -> FloatArray:
        """``sqrt(x^T A_k^-1 x)`` for every arm: the LinUCB confidence width."""
        vector = np.asarray(features, dtype=np.float64)
        out = np.empty(self.n_arms, dtype=np.float64)
        for arm in range(self.n_arms):
            projected = self._ensure_inverse(arm) @ vector
            out[arm] = max(0.0, float(vector @ projected))
        return np.sqrt(out)

    def means_matrix(self, features: FloatArray) -> FloatArray:
        """Predicted rewards for a batch, shape ``(n, n_arms)``."""
        matrix = np.asarray(features, dtype=np.float64)
        return matrix @ self.theta_matrix().T

    def widths_matrix(self, features: FloatArray) -> FloatArray:
        """Confidence widths for a batch, shape ``(n, n_arms)``."""
        matrix = np.asarray(features, dtype=np.float64)
        out = np.empty((matrix.shape[0], self.n_arms), dtype=np.float64)
        for arm in range(self.n_arms):
            projected = matrix @ self._ensure_inverse(arm)
            out[:, arm] = np.maximum(0.0, np.einsum("ij,ij->i", projected, matrix))
        return np.sqrt(out)

    # --- serialization --------------------------------------------------------------------

    def arrays(self) -> dict[str, FloatArray]:
        """Sufficient statistics, for a pickle-free snapshot."""
        return {
            "gram": self._gram.copy(),
            "rhs": self._rhs.copy(),
            "mass": self._mass.copy(),
        }

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, FloatArray],
        *,
        ridge: float,
        discount: float,
        refresh_every: int = 2_048,
    ) -> DiscountedRidge:
        """Rebuild posteriors from :meth:`arrays` output."""
        gram = np.asarray(arrays["gram"], dtype=np.float64)
        rhs = np.asarray(arrays["rhs"], dtype=np.float64)
        mass = np.asarray(arrays["mass"], dtype=np.float64)
        if gram.ndim != 3 or gram.shape[1] != gram.shape[2]:
            raise ValueError("gram must have shape (n_arms, dim, dim)")
        if rhs.shape != gram.shape[:2] or mass.shape != gram.shape[:1]:
            raise ValueError("rhs and mass shapes do not match gram")
        posterior = cls(
            gram.shape[0],
            gram.shape[1],
            ridge=ridge,
            discount=discount,
            refresh_every=refresh_every,
        )
        posterior._gram = gram.copy()
        posterior._rhs = rhs.copy()
        posterior._mass = mass.copy()
        posterior._inverse_fresh[:] = False
        posterior._theta_fresh[:] = False
        return posterior
