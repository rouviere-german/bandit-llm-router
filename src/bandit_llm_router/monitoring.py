"""Context drift monitoring for the logged features.

A learned router is trained on the traffic it saw. When the traffic changes -- a new tenant, a new
prompt template, a client that suddenly sends 10x longer inputs -- the policy keeps acting
confidently on contexts it has no evidence about, and the off-policy estimates that screened it were
computed on a distribution that no longer exists. Drift in the *inputs* is the earliest signal
available, and it needs no labels, so it is worth watching continuously.

This wraps the ML kit's PSI and KS monitors over the named feature columns of the decision log and
exposes the per-feature PSI as the ``context_feature_psi{feature}`` gauge. The usual PSI reading
applies: below 0.1 stable, 0.1-0.2 moderate, above 0.2 a real shift. Constant columns (the bias
term) report 0 by construction.

What this does not do: it says nothing about whether the *reward* relationship changed. That is what
the scheduled-shift experiment and the discounted posteriors are for, and a drift alarm here is a
reason to re-screen, not an automatic rollback.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from bandit_llm_router.ml.drift import DriftReport, psi

__all__ = ["ContextDriftMonitor"]

FloatArray = npt.NDArray[np.float64]


class ContextDriftMonitor:
    """Compare a window of logged feature rows against a reference window, column by column."""

    def __init__(
        self,
        reference: FloatArray,
        feature_names: Sequence[str],
        *,
        psi_threshold: float = 0.2,
        bins: int = 10,
    ) -> None:
        matrix = np.asarray(reference, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("reference must be a 2-D array with at least two rows")
        if len(feature_names) != matrix.shape[1]:
            raise ValueError("feature_names must name every reference column")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("reference contains NaN or infinite values")
        if psi_threshold <= 0.0:
            raise ValueError("psi_threshold must be positive")
        self._reference = matrix
        self._names = tuple(feature_names)
        self.psi_threshold = psi_threshold
        self.bins = bins

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def psi_by_feature(self, current: FloatArray) -> dict[str, float]:
        """PSI of every feature column, keyed by feature name."""
        matrix = np.asarray(current, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self._reference.shape[1]:
            raise ValueError("current must have the same number of columns as the reference")
        if matrix.shape[0] < 2:
            raise ValueError("current needs at least two rows")
        return {
            name: psi(self._reference[:, column], matrix[:, column], bins=self.bins)
            for column, name in enumerate(self._names)
        }

    def drifted_features(self, current: FloatArray) -> tuple[str, ...]:
        scores = self.psi_by_feature(current)
        return tuple(name for name, score in scores.items() if score > self.psi_threshold)

    def check(self, current: FloatArray) -> DriftReport:
        """Full PSI + KS report, using the ML kit's monitor semantics."""
        from bandit_llm_router.ml.drift import DriftMonitor  # noqa: PLC0415 - avoids a cycle

        matrix = np.asarray(current, dtype=np.float64)
        monitor = DriftMonitor(
            {name: self._reference[:, column] for column, name in enumerate(self._names)},
            psi_threshold=self.psi_threshold,
            bins=self.bins,
        )
        return monitor.check({name: matrix[:, column] for column, name in enumerate(self._names)})
