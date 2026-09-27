"""Policy activation: screened, approved by a second principal, and hash-chain audited.

Activating a policy changes what every future request costs and how good it is, so the gate is the
most security-relevant code in the repo. Each of the three requirements is tested by trying to get
past it:

* an ineligible candidate cannot even be proposed (and the refusal is audited);
* the proposer cannot approve their own proposal;
* a report swapped between approval and activation is refused, because the stored evidence is
  re-hashed and compared with the digest recorded at proposal time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from bandit_llm_router.activation import (
    ACTIVE_ALIAS,
    POLICY_MODEL_NAME,
    ActivationError,
    ActivationRecord,
    ActivationStatus,
    NotEligibleError,
    PolicyActivationController,
    SeparationOfDutiesError,
    StaleReportError,
    UnknownActivationError,
)
from bandit_llm_router.agentic.audit import AuditLog, verify_file
from bandit_llm_router.ml.registry import ModelRegistry, RegistryError
from bandit_llm_router.ope.screening import ScreeningReport
from bandit_llm_router.policies.linear_bandit import LinUCB

pytestmark = pytest.mark.unit

ARMS = ("haiku", "sonnet", "opus")
DIM = 5
PROPOSER = "admin-key-01"
APPROVER = "approve-key-9"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture
def controller(tmp_path: Path) -> PolicyActivationController:
    ids = count(1)
    return PolicyActivationController(
        registry=ModelRegistry(tmp_path / "registry"),
        audit=AuditLog(tmp_path / "activations.audit.jsonl"),
        path=tmp_path / "activations.json",
        clock=FakeClock(),
        id_factory=lambda: f"{next(ids):032x}",
    )


def policy(alpha: float = 0.5) -> LinUCB:
    built = LinUCB(arms=ARMS, dim=DIM, alpha=alpha, feature_version="ctx-test")
    rng = np.random.default_rng(int(alpha * 100))
    for step in range(20):
        built.update(rng.normal(size=DIM) * 0.2, step % 3, float(rng.random()))
    return built


def report(
    candidate_version: str, *, eligible: bool = True, logs_digest: str | None = None
) -> ScreeningReport:
    """A hand-built report, so the gate can be tested without running a whole screening."""
    estimate = {
        "method": "dr",
        "value": 0.62,
        "std_error": 0.01,
        "ci_low": 0.60,
        "ci_high": 0.64,
        "ess": 900.0,
        "clipped_fraction": 0.0,
    }
    return ScreeningReport.model_validate(
        {
            "created_at": "2026-01-01T00:00:00+00:00",
            "candidate_version": candidate_version,
            "candidate_name": "linucb",
            "incumbent_version": "linucb-incumbent",
            "logs_digest": logs_digest or ("c" * 64),
            "config": {},
            "n": 5_000,
            "ess": 900.0,
            "min_propensity": 0.02,
            "mean_logged_reward": 0.5,
            "candidate_ips": {**estimate, "method": "ips"},
            "candidate_snips": {**estimate, "method": "snips"},
            "candidate_dr": estimate,
            "incumbent_dr": {**estimate, "value": 0.55, "ci_low": 0.53, "ci_high": 0.57},
            "dr_lower_bound": 0.60,
            "improvement": 0.05,
            "cost": {
                "expected_usd_per_request": 0.002,
                "worst_case_usd_per_request": 0.004,
                "worst_case_horizon_usd": 40.0,
            },
            "eligible": eligible,
            "reasons": () if eligible else ("dr_lower_bound_does_not_beat_incumbent",),
        }
    )


def register(controller: PolicyActivationController, candidate: LinUCB) -> str:
    return controller.register_snapshot(candidate.snapshot(), seed=1).digest


def promote(
    controller: PolicyActivationController, candidate: LinUCB
) -> tuple[str, ActivationRecord]:
    digest = register(controller, candidate)
    record = controller.propose(
        snapshot_digest=digest, report=report(candidate.version), proposed_by=PROPOSER
    )
    return digest, record


# --- the registry --------------------------------------------------------------------------------


def test_a_snapshot_is_stored_without_pickle(controller: PolicyActivationController) -> None:
    candidate = policy()
    version = controller.register_snapshot(candidate.snapshot(), seed=7)
    assert set(version.files) == {"params.json", "posterior.npz"}
    assert version.manifest.params["policy_version"] == candidate.version
    assert controller.active_version() is None


def test_the_active_alias_moves_only_through_the_gate(
    controller: PolicyActivationController,
) -> None:
    candidate = policy()
    digest, record = promote(controller, candidate)
    approved = controller.approve(record.id, approver=APPROVER)
    activated = controller.activate(approved.id, actor=PROPOSER)
    assert activated.status is ActivationStatus.ACTIVATED
    active = controller.active_version()
    assert active is not None
    assert active.digest == digest
    assert digest in controller.records()[0].snapshot_digest


# --- gate 1: screening ---------------------------------------------------------------------------


def test_an_ineligible_candidate_cannot_be_proposed(
    controller: PolicyActivationController,
) -> None:
    candidate = policy()
    digest = register(controller, candidate)
    with pytest.raises(NotEligibleError, match="failed OPE screening"):
        controller.propose(
            snapshot_digest=digest,
            report=report(candidate.version, eligible=False),
            proposed_by=PROPOSER,
        )
    assert controller.records() == ()


def test_a_refused_proposal_is_still_audited(controller: PolicyActivationController) -> None:
    """A rejection is evidence too: an operator retrying until one passes must be visible."""
    candidate = policy()
    digest = register(controller, candidate)
    head_before = controller._audit.head
    with pytest.raises(NotEligibleError):
        controller.propose(
            snapshot_digest=digest,
            report=report(candidate.version, eligible=False),
            proposed_by=PROPOSER,
        )
    assert controller._audit.head != head_before
    assert verify_file(controller._audit.path).ok


def test_a_report_for_a_different_snapshot_is_refused(
    controller: PolicyActivationController,
) -> None:
    digest = register(controller, policy(alpha=0.5))
    other = policy(alpha=0.9)
    with pytest.raises(ActivationError, match="but the snapshot holds"):
        controller.propose(
            snapshot_digest=digest, report=report(other.version), proposed_by=PROPOSER
        )


def test_proposing_an_unregistered_snapshot_fails(
    controller: PolicyActivationController,
) -> None:
    with pytest.raises(RegistryError):
        controller.propose(
            snapshot_digest="f" * 64, report=report("linucb-whatever"), proposed_by=PROPOSER
        )


# --- gate 2: a second principal -------------------------------------------------------------------


def test_the_proposer_cannot_approve_their_own_activation(
    controller: PolicyActivationController,
) -> None:
    _, record = promote(controller, policy())
    with pytest.raises(SeparationOfDutiesError, match="separation of duties"):
        controller.approve(record.id, approver=PROPOSER)
    assert controller.get(record.id).status is ActivationStatus.PROPOSED


def test_activation_needs_an_approval_first(controller: PolicyActivationController) -> None:
    _, record = promote(controller, policy())
    with pytest.raises(ActivationError, match="is proposed, not approved"):
        controller.activate(record.id, actor=PROPOSER)
    assert controller.active_version() is None


def test_an_approval_cannot_be_replayed(controller: PolicyActivationController) -> None:
    _, record = promote(controller, policy())
    controller.approve(record.id, approver=APPROVER)
    with pytest.raises(ActivationError, match="is approved, not proposed"):
        controller.approve(record.id, approver=APPROVER)


def test_an_activation_cannot_be_replayed(controller: PolicyActivationController) -> None:
    _, record = promote(controller, policy())
    controller.approve(record.id, approver=APPROVER)
    controller.activate(record.id, actor=PROPOSER)
    with pytest.raises(ActivationError, match="is activated, not approved"):
        controller.activate(record.id, actor=PROPOSER)


def test_a_human_may_reject_an_eligible_candidate(
    controller: PolicyActivationController,
) -> None:
    _, record = promote(controller, policy())
    rejected = controller.reject(record.id, actor=APPROVER, reason="waiting for the load test")
    assert rejected.status is ActivationStatus.REJECTED
    assert rejected.reason == "waiting for the load test"
    with pytest.raises(ActivationError, match="not proposed"):
        controller.approve(record.id, approver=APPROVER)


def test_an_activated_policy_cannot_be_rejected_retroactively(
    controller: PolicyActivationController,
) -> None:
    _, record = promote(controller, policy())
    controller.approve(record.id, approver=APPROVER)
    controller.activate(record.id, actor=PROPOSER)
    with pytest.raises(ActivationError, match="retroactively"):
        controller.reject(record.id, actor=APPROVER, reason="changed my mind")


def test_an_unknown_activation_id_is_refused(controller: PolicyActivationController) -> None:
    unknown = "0" * 32
    calls: tuple[Callable[[], ActivationRecord], ...] = (
        lambda: controller.get(unknown),
        lambda: controller.approve(unknown, approver=APPROVER),
        lambda: controller.activate(unknown, actor=PROPOSER),
        lambda: controller.reject(unknown, actor=APPROVER, reason="x"),
    )
    for call in calls:
        with pytest.raises(UnknownActivationError):
            call()


# --- gate 3: the evidence still matches -----------------------------------------------------------


def rewrite_store(path: Path, record: ActivationRecord) -> None:
    """Edit the persisted activation store the way an attacker with file access would."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"] = [
        record.model_dump(mode="json") if item["id"] == record.id else item
        for item in payload["records"]
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_report_swapped_after_approval_is_refused(
    controller: PolicyActivationController, tmp_path: Path
) -> None:
    """Otherwise an approval could be moved onto a log where the candidate happens to look good."""
    candidate = policy()
    _, record = promote(controller, candidate)
    approved = controller.approve(record.id, approver=APPROVER)
    rewrite_store(
        tmp_path / "activations.json",
        approved.model_copy(update={"report": report(candidate.version, logs_digest="d" * 64)}),
    )
    with pytest.raises(StaleReportError, match="no longer matches its digest"):
        controller.activate(approved.id, actor=PROPOSER)
    assert controller.active_version() is None


def test_a_report_for_another_candidate_version_is_refused(
    controller: PolicyActivationController, tmp_path: Path
) -> None:
    candidate = policy()
    _, record = promote(controller, candidate)
    rewrite_store(
        tmp_path / "activations.json",
        record.model_copy(update={"candidate_version": "linucb-somethingelse"}),
    )
    with pytest.raises(StaleReportError, match="different candidate version"):
        controller.approve(record.id, approver=APPROVER)


# --- the audit chain ------------------------------------------------------------------------------


def test_every_step_is_appended_to_a_verifiable_chain(
    controller: PolicyActivationController, tmp_path: Path
) -> None:
    candidate = policy()
    _, record = promote(controller, candidate)
    controller.approve(record.id, approver=APPROVER)
    controller.activate(record.id, actor=PROPOSER)
    path = tmp_path / "activations.audit.jsonl"
    result = verify_file(path, expect_head=controller._audit.head)
    assert result.ok
    assert result.entries == 3
    actions = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert "policy_activation_proposed" in actions[0]
    assert "policy_activation_approved" in actions[1]
    assert "policy_activated" in actions[2]


def test_editing_the_chain_breaks_verification(
    controller: PolicyActivationController, tmp_path: Path
) -> None:
    _, record = promote(controller, policy())
    controller.approve(record.id, approver=APPROVER)
    path = tmp_path / "activations.audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:1]) + "\n", encoding="utf-8")
    assert not verify_file(path, expect_head=controller._audit.head).ok


def test_the_audit_payload_carries_the_evidence_not_the_prompt(
    controller: PolicyActivationController,
) -> None:
    candidate = policy()
    _, record = promote(controller, candidate)
    payload = record.audit_payload()
    assert payload["candidate_version"] == candidate.version
    assert payload["proposed_by"] == PROPOSER
    assert payload["dr_lower_bound"] == 0.60
    assert payload["logs_digest"] == "c" * 64
    assert set(payload) == {
        "activation_id",
        "candidate_version",
        "snapshot_digest",
        "report_digest",
        "logs_digest",
        "dr_lower_bound",
        "incumbent_dr",
        "ess",
        "worst_case_usd",
        "proposed_by",
        "approved_by",
    }


# --- persistence ---------------------------------------------------------------------------------


def test_the_activation_history_survives_a_restart(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    audit = AuditLog(tmp_path / "audit.jsonl")
    store = tmp_path / "activations.json"
    ids = count(1)
    first = PolicyActivationController(
        registry=registry,
        audit=audit,
        path=store,
        clock=FakeClock(),
        id_factory=lambda: f"{next(ids):032x}",
    )
    candidate = policy()
    _, record = promote(first, candidate)
    restarted = PolicyActivationController(registry=registry, audit=audit, path=store)
    reloaded = restarted.get(record.id)
    assert reloaded.candidate_version == candidate.version
    assert reloaded.status is ActivationStatus.PROPOSED
    # A different process can approve it, which is the point of persisting.
    assert restarted.approve(record.id, approver=APPROVER).approved_by == APPROVER


def test_a_corrupt_activation_store_fails_loudly(tmp_path: Path) -> None:
    store = tmp_path / "activations.json"
    store.write_text('{"version": 1, "records": [{"id": "nope"}]}', encoding="utf-8")
    with pytest.raises(ActivationError, match="failed validation"):
        PolicyActivationController(
            registry=ModelRegistry(tmp_path / "registry"),
            audit=AuditLog(tmp_path / "audit.jsonl"),
            path=store,
        )


def test_the_history_digest_anchors_the_whole_record(
    controller: PolicyActivationController,
) -> None:
    empty = controller.history_digest()
    _, record = promote(controller, policy())
    proposed = controller.history_digest()
    assert proposed != empty
    controller.approve(record.id, approver=APPROVER)
    assert controller.history_digest() != proposed


def test_an_in_memory_controller_needs_no_files(tmp_path: Path) -> None:
    controller = PolicyActivationController(
        registry=ModelRegistry(tmp_path / "registry"),
        audit=AuditLog(tmp_path / "audit.jsonl"),
        path=None,
        clock=FakeClock(),
        id_factory=lambda: "a" * 32,
    )
    candidate = policy()
    digest, record = promote(controller, candidate)
    assert controller.get(record.id).snapshot_digest == digest
    assert not (tmp_path / "activations.json").exists()


def test_the_model_and_alias_names_are_fixed_constants() -> None:
    assert POLICY_MODEL_NAME == "router-policy"
    assert ACTIVE_ALIAS == "active"
