"""Policy activation: screened, approved by a second person, and hash-chain audited.

Activating a policy is the one action in this system that changes what every future request costs
and how good it is. It therefore takes three independent things, and the order is enforced:

1. **A passing screening report.** :func:`~bandit_llm_router.ope.screening.screen_candidate` must
have
   marked the candidate eligible on a specific log. A proposal carries the whole report, and the
   report's content digest is stored with it.
2. **An approval from a different principal.** The proposer cannot approve their own proposal
   (``admin`` and ``approve`` are separate scopes for exactly this reason).
3. **A matching, untampered record at activation time.** Before the alias moves, the stored report
is
   re-hashed and compared with the digest recorded at proposal time, and the candidate version in
   the report must still match the registered snapshot. A report swapped after approval -- for a log
   where the candidate happens to look better -- is refused.

Every step appends to a hash-chained audit log (``proposed``, ``approved``, ``activated``,
``rejected``), so the history of who promoted what, on which evidence, is tamper-evident: editing or
removing an entry breaks the chain and ``bandit-llm-router audit verify`` fails.

Policy parameters themselves live in the content-addressed model registry (arrays as ``.npz`` with
``allow_pickle=False``, parameters as JSON -- no pickle anywhere), and the alias ``active`` points
at the version currently serving. The registry verifies a version record's hash on load, so a
snapshot cannot be edited underneath the alias.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from bandit_llm_router.agentic.audit import AuditLog
from bandit_llm_router.core.canonical import canonical_json, sha256_hex
from bandit_llm_router.core.fileio import atomic_write_text, file_lock
from bandit_llm_router.ml.manifest import RunManifest, create_manifest
from bandit_llm_router.ml.registry import ModelRegistry, ModelVersion
from bandit_llm_router.ml.serialization import dumps_npz
from bandit_llm_router.ope.screening import ScreeningReport
from bandit_llm_router.policies.base import PolicySnapshot

__all__ = [
    "ACTIVE_ALIAS",
    "POLICY_MODEL_NAME",
    "ActivationError",
    "ActivationRecord",
    "ActivationStatus",
    "NotEligibleError",
    "PolicyActivationController",
    "SeparationOfDutiesError",
    "StaleReportError",
    "UnknownActivationError",
]

POLICY_MODEL_NAME: Final = "router-policy"
ACTIVE_ALIAS: Final = "active"


class ActivationError(RuntimeError):
    """An activation request was refused."""


class NotEligibleError(ActivationError):
    """The screening report did not clear the gates, so the candidate cannot be proposed."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        super().__init__("candidate failed OPE screening: " + (", ".join(reasons) or "unknown"))
        self.reasons = reasons


class SeparationOfDutiesError(ActivationError):
    """The approver must not be the principal that proposed the activation."""


class UnknownActivationError(ActivationError):
    """No activation with that id exists."""


class StaleReportError(ActivationError):
    """The stored evidence no longer matches what was approved."""


class ActivationStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVATED = "activated"
    REJECTED = "rejected"


class ActivationRecord(BaseModel):
    """One activation request, from proposal to outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    candidate_version: str
    snapshot_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    report: ScreeningReport
    proposed_by: str = Field(min_length=1, max_length=64)
    proposed_at: float
    status: ActivationStatus = ActivationStatus.PROPOSED
    approved_by: str | None = None
    approved_at: float | None = None
    activated_at: float | None = None
    reason: str | None = None

    def audit_payload(self) -> dict[str, JsonValue]:
        """The non-secret facts an auditor needs, small enough to keep in the chain."""
        return {
            "activation_id": self.id,
            "candidate_version": self.candidate_version,
            "snapshot_digest": self.snapshot_digest,
            "report_digest": self.report_digest,
            "logs_digest": self.report.logs_digest,
            "dr_lower_bound": self.report.dr_lower_bound,
            "incumbent_dr": self.report.incumbent_dr.value,
            "ess": self.report.ess,
            "horizon_usd": self.report.cost.upper_bound_horizon_usd,
            "proposed_by": self.proposed_by,
            "approved_by": self.approved_by,
        }


class _Store(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    records: list[ActivationRecord] = Field(default_factory=list)


class PolicyActivationController:
    """Registers policy snapshots and gates their promotion to ``active``."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audit: AuditLog,
        path: Path | None = None,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._path = path
        self._clock = clock
        self._id_factory = id_factory
        self._lock = threading.Lock()
        self._records: dict[str, ActivationRecord] = {}
        with self._lock:
            self._reload()

    # --- persistence ----------------------------------------------------------------------

    def _reload(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            stored = _Store.model_validate_json(self._path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ActivationError(f"activation store {self._path.name} failed validation") from exc
        self._records = {record.id: record for record in stored.records}

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = _Store(records=list(self._records.values())).model_dump(mode="json")
        atomic_write_text(self._path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _sync(self) -> None:
        if self._path is not None:
            with file_lock(self._path):
                self._reload()

    # --- registry -------------------------------------------------------------------------

    def register_snapshot(
        self,
        snapshot: PolicySnapshot,
        *,
        seed: int = 0,
        metrics: Mapping[str, float] | None = None,
        manifest: RunManifest | None = None,
    ) -> ModelVersion:
        """Store a policy snapshot in the content-addressed registry (JSON + npz, no pickle)."""
        params = dict(snapshot.params)
        files = {
            "params.json": (json.dumps(params, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            "posterior.npz": dumps_npz(dict(snapshot.arrays)),
        }
        run_manifest = manifest or create_manifest(
            seed=seed,
            datasets=(),
            params={"policy_version": snapshot.version, "policy_name": snapshot.name},
            metrics=dict(metrics or {}),
        )
        return self._registry.register(
            POLICY_MODEL_NAME, files, run_manifest, metrics=dict(metrics or {})
        )

    def active_version(self) -> ModelVersion | None:
        """The snapshot the ``active`` alias points at, verified on load."""
        try:
            return self._registry.get(POLICY_MODEL_NAME, ACTIVE_ALIAS)
        except ValueError:
            return None

    # --- workflow -------------------------------------------------------------------------

    def records(self) -> tuple[ActivationRecord, ...]:
        self._sync()
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda record: record.proposed_at))

    def get(self, activation_id: str) -> ActivationRecord:
        self._sync()
        with self._lock:
            record = self._records.get(activation_id)
        if record is None:
            raise UnknownActivationError(f"unknown activation {activation_id}")
        return record

    def propose(
        self,
        *,
        snapshot_digest: str,
        report: ScreeningReport,
        proposed_by: str,
    ) -> ActivationRecord:
        """Record a proposal for an eligible candidate. Refuses ineligible reports outright."""
        version = self._registry.get(POLICY_MODEL_NAME, snapshot_digest)
        registered_version = str(version.manifest.params.get("policy_version", ""))
        if registered_version != report.candidate_version:
            raise ActivationError(
                f"report is for {report.candidate_version} but the snapshot holds "
                f"{registered_version or 'an unknown version'}"
            )
        if not report.eligible:
            self._audit.append(
                proposed_by,
                "policy_activation_rejected",
                {
                    "candidate_version": report.candidate_version,
                    "snapshot_digest": snapshot_digest,
                    "reasons": list(report.reasons),
                },
            )
            raise NotEligibleError(report.reasons)
        record = ActivationRecord(
            id=self._id_factory(),
            candidate_version=report.candidate_version,
            snapshot_digest=snapshot_digest,
            report_digest=report.digest(),
            report=report,
            proposed_by=proposed_by,
            proposed_at=self._clock(),
        )
        with self._lock:
            self._sync_locked()
            self._records[record.id] = record
            self._persist()
        self._audit.append(proposed_by, "policy_activation_proposed", record.audit_payload())
        return record

    def _sync_locked(self) -> None:
        if self._path is not None:
            self._reload()

    def approve(self, activation_id: str, *, approver: str) -> ActivationRecord:
        """Approve a proposal. The proposer may not approve their own."""
        with self._lock:
            self._sync_locked()
            record = self._records.get(activation_id)
            if record is None:
                raise UnknownActivationError(f"unknown activation {activation_id}")
            if record.status is not ActivationStatus.PROPOSED:
                raise ActivationError(
                    f"activation {activation_id} is {record.status.value}, not proposed"
                )
            if approver == record.proposed_by:
                raise SeparationOfDutiesError(
                    "separation of duties: the proposer cannot approve their own activation"
                )
            self._verify_report(record)
            approved = record.model_copy(
                update={
                    "status": ActivationStatus.APPROVED,
                    "approved_by": approver,
                    "approved_at": self._clock(),
                }
            )
            self._records[activation_id] = approved
            self._persist()
        self._audit.append(approver, "policy_activation_approved", approved.audit_payload())
        return approved

    def reject(self, activation_id: str, *, actor: str, reason: str) -> ActivationRecord:
        """Reject a proposal (a human may refuse an eligible candidate)."""
        with self._lock:
            self._sync_locked()
            record = self._records.get(activation_id)
            if record is None:
                raise UnknownActivationError(f"unknown activation {activation_id}")
            if record.status is ActivationStatus.ACTIVATED:
                raise ActivationError("an activated policy cannot be rejected retroactively")
            rejected = record.model_copy(
                update={"status": ActivationStatus.REJECTED, "reason": reason[:500]}
            )
            self._records[activation_id] = rejected
            self._persist()
        self._audit.append(
            actor,
            "policy_activation_rejected",
            {**rejected.audit_payload(), "reason": rejected.reason},
        )
        return rejected

    def activate(self, activation_id: str, *, actor: str) -> ActivationRecord:
        """Move the ``active`` alias, but only for an approved, still-matching proposal."""
        with self._lock:
            self._sync_locked()
            record = self._records.get(activation_id)
            if record is None:
                raise UnknownActivationError(f"unknown activation {activation_id}")
            if record.status is not ActivationStatus.APPROVED:
                raise ActivationError(
                    f"activation {activation_id} is {record.status.value}, not approved"
                )
            self._verify_report(record)
            self._registry.set_alias(POLICY_MODEL_NAME, ACTIVE_ALIAS, record.snapshot_digest)
            activated = record.model_copy(
                update={"status": ActivationStatus.ACTIVATED, "activated_at": self._clock()}
            )
            self._records[activation_id] = activated
            self._persist()
        self._audit.append(actor, "policy_activated", activated.audit_payload())
        return activated

    def _verify_report(self, record: ActivationRecord) -> None:
        """Re-hash the stored report and refuse a record whose evidence changed."""
        if record.report.digest() != record.report_digest:
            raise StaleReportError(
                f"activation {record.id}: the stored screening report no longer matches its digest"
            )
        if record.report.candidate_version != record.candidate_version:
            raise StaleReportError(
                f"activation {record.id}: the report is for a different candidate version"
            )

    # --- reporting ------------------------------------------------------------------------

    def history_digest(self) -> str:
        """Hash of the activation history, an anchor to pair with the audit log's head."""
        return sha256_hex(
            canonical_json(
                [record.model_dump(mode="json") for record in self.records()],
            )
        )
