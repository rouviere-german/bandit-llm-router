"""Per-tenant USD caps, enforced before a provider call rather than noticed afterwards.

The guard is built around one rule: **a call is admitted only if its worst-case cost still fits the
cap**. The worst case is the arm's cost ceiling -- the request's ``max_output_tokens`` (capped by
the arm's own limit) at that arm's prices -- so the amount committed after the call can never exceed
the amount reserved before it. That is what makes ``spent <= cap`` an invariant instead of an
aspiration, and it is what ``tests/unit/test_budgets.py`` proves with a property test over random
sequences of reservations, commits and releases.

Three consequences worth stating:

* **Exploration is priced.** An arm the policy would like to try is simply unavailable when its
  ceiling does not fit; its probability mass moves to an affordable arm and the *realized*
  propensity is logged (the same redistribution the circuit breaker uses). Exploration therefore
  cannot push a tenant over its cap, which is the property the OPE screening gate also relies on.
* **Reservations are released.** A failed provider call refunds its reservation, so an outage does
  not silently consume a tenant's budget.
* **Windows roll over.** Each tenant's spend accumulates over ``window_seconds`` (a day by default)
  and resets when the window rolls. The state is a small JSON file written atomically under a file
  lock, so several workers share one budget instead of each spending the full amount.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bandit_llm_router.core.fileio import atomic_write_text, file_lock

__all__ = [
    "BudgetExhaustedError",
    "TenantBudgetError",
    "TenantBudgetGuard",
    "TenantBudgets",
    "TenantSpend",
]

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
_MICROS_PER_USD = 1_000_000


class TenantBudgetError(RuntimeError):
    """Budget state or a budget operation was invalid."""


class BudgetExhaustedError(TenantBudgetError):
    """The tenant cannot afford this call's worst case."""

    def __init__(self, tenant: str, requested_micros: int, remaining_micros: int) -> None:
        super().__init__(
            f"tenant {tenant} has {remaining_micros} micro-USD left and needs {requested_micros}"
        )
        self.tenant = tenant
        self.requested_micros = requested_micros
        self.remaining_micros = remaining_micros


class TenantBudgets(BaseModel):
    """Caps per tenant, in USD per window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    caps_usd: Mapping[str, float] = Field(default_factory=dict)
    default_usd: float = Field(default=5.0, ge=0.0)
    window_seconds: float = Field(default=86_400.0, gt=0.0)

    def cap_micros(self, tenant: str) -> int:
        usd = self.caps_usd.get(tenant, self.default_usd)
        if usd < 0.0:
            raise TenantBudgetError(f"negative cap configured for tenant {tenant}")
        return round(usd * _MICROS_PER_USD)

    def digest_payload(self) -> dict[str, object]:
        return {
            "caps_usd": {name: self.caps_usd[name] for name in sorted(self.caps_usd)},
            "default_usd": self.default_usd,
            "window_seconds": self.window_seconds,
        }


class TenantSpend(BaseModel):
    """One tenant's committed and reserved spend inside the current window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_start: float
    spent_micros: int = Field(default=0, ge=0)
    reserved_micros: int = Field(default=0, ge=0)

    @property
    def committed_and_reserved(self) -> int:
        return self.spent_micros + self.reserved_micros


class _GuardFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    tenants: dict[str, TenantSpend] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Reservation:
    """A held amount for one in-flight call."""

    tenant: str
    arm: str
    micros: int


class TenantBudgetGuard:
    """Reserve-before-call, commit-after-call spend caps per tenant."""

    def __init__(
        self,
        budgets: TenantBudgets | None = None,
        *,
        path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._budgets = budgets if budgets is not None else TenantBudgets()
        self._path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, TenantSpend] = {}
        with self._lock:
            self._reload()

    @property
    def budgets(self) -> TenantBudgets:
        return self._budgets

    # --- persistence ----------------------------------------------------------------------

    def _reload(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            stored = _GuardFile.model_validate_json(self._path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise TenantBudgetError(f"budget file {self._path.name} failed validation") from exc
        self._state = dict(stored.tenants)

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = _GuardFile(tenants=self._state).model_dump(mode="json")
        atomic_write_text(self._path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _guard(self) -> None:
        """Called with the thread lock held: refresh from disk so workers share one budget."""
        if self._path is not None:
            self._reload()

    def _current(self, tenant: str) -> TenantSpend:
        now = self._clock()
        spend = self._state.get(tenant)
        if spend is None or now - spend.window_start >= self._budgets.window_seconds:
            spend = TenantSpend(window_start=now)
            self._state[tenant] = spend
        return spend

    # --- queries --------------------------------------------------------------------------

    def spent_micros(self, tenant: str) -> int:
        with self._lock, self._file_lock():
            self._guard()
            return self._current(tenant).spent_micros

    def remaining_micros(self, tenant: str) -> int:
        with self._lock, self._file_lock():
            self._guard()
            spend = self._current(tenant)
            return max(0, self._budgets.cap_micros(tenant) - spend.committed_and_reserved)

    def affordable_mask(self, tenant: str, ceilings: FloatArray) -> BoolArray:
        """Which arms' worst-case cost still fits the tenant's remaining budget."""
        remaining = self.remaining_micros(tenant)
        return np.asarray(ceilings, dtype=np.float64) <= float(remaining)

    # --- mutations ------------------------------------------------------------------------

    def _file_lock(self) -> contextlib.AbstractContextManager[None]:
        """Cross-process lock for a persisted guard; a no-op for an in-memory one."""
        if self._path is None:
            return contextlib.nullcontext()
        return file_lock(self._path)

    def reserve(self, tenant: str, arm: str, ceiling_micros: int) -> Reservation:
        """Hold ``ceiling_micros`` for an in-flight call, or refuse the call."""
        if ceiling_micros < 0:
            raise ValueError("ceiling_micros must be non-negative")
        with self._lock, self._file_lock():
            self._guard()
            spend = self._current(tenant)
            cap = self._budgets.cap_micros(tenant)
            if spend.committed_and_reserved + ceiling_micros > cap:
                raise BudgetExhaustedError(
                    tenant, ceiling_micros, max(0, cap - spend.committed_and_reserved)
                )
            self._state[tenant] = spend.model_copy(
                update={"reserved_micros": spend.reserved_micros + ceiling_micros}
            )
            self._persist()
        return Reservation(tenant=tenant, arm=arm, micros=ceiling_micros)

    def commit(self, reservation: Reservation, actual_micros: int) -> int:
        """Settle a reservation with the real cost. Returns the committed amount."""
        if actual_micros < 0:
            raise ValueError("actual_micros must be non-negative")
        charged = min(actual_micros, reservation.micros)
        with self._lock, self._file_lock():
            self._guard()
            spend = self._current(reservation.tenant)
            if spend.reserved_micros < reservation.micros:
                # The window rolled over while the call was in flight; charge without refunding a
                # reservation that no longer exists in this window.
                self._state[reservation.tenant] = spend.model_copy(
                    update={
                        "spent_micros": spend.spent_micros + charged,
                        "reserved_micros": 0,
                    }
                )
            else:
                self._state[reservation.tenant] = spend.model_copy(
                    update={
                        "spent_micros": spend.spent_micros + charged,
                        "reserved_micros": spend.reserved_micros - reservation.micros,
                    }
                )
            self._persist()
        return charged

    def release(self, reservation: Reservation) -> None:
        """Give a reservation back (the call failed, so nothing was billed)."""
        self.commit(reservation, 0)

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Per-tenant spent / reserved / remaining, for metrics and the demo tables."""
        with self._lock, self._file_lock():
            self._guard()
            out: dict[str, dict[str, int]] = {}
            for tenant in sorted(self._state):
                spend = self._current(tenant)
                cap = self._budgets.cap_micros(tenant)
                out[tenant] = {
                    "spent_micros": spend.spent_micros,
                    "reserved_micros": spend.reserved_micros,
                    "remaining_micros": max(0, cap - spend.committed_and_reserved),
                    "cap_micros": cap,
                }
            return out
