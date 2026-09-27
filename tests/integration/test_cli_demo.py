"""The CLI and the offline demo.

The demo is the repo's shortest honest tour, so the test asserts the numbers it prints are the ones
the README claims -- cost saving, sublinear in-class regret, OPE accuracy, a real recovery after the
shift, an activation that actually happened, and a verified audit chain. It runs the demo once at a
reduced size; ``uv run python tasks.py demo`` runs it at full size.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from bandit_llm_router import cli, demo
from bandit_llm_router.agentic.audit import AuditLog
from bandit_llm_router.core import tracing
from bandit_llm_router.core.security import hash_api_key

pytestmark = pytest.mark.integration


@pytest.fixture
def small_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink every knob so the whole demo runs inside a test's budget.

    The sizes change; the code paths do not. The shift step follows ``_REQUESTS``, so the mid-run
    quality shift still lands halfway through.
    """
    monkeypatch.setattr(demo, "_REQUESTS", 6_000)
    monkeypatch.setattr(demo, "_LOG_REQUESTS", 3_000)
    monkeypatch.setattr(demo, "_WARMUP", 800)
    monkeypatch.setattr(demo, "_POISON_REQUESTS", 2_000)
    monkeypatch.setattr(demo, "_BOOTSTRAP", 80)


def check_governance(activation: object) -> None:
    """A candidate was screened, approved by a second principal and activated -- and rightly so.

    The second half is the part worth having: the demo reports each candidate's *true* policy value
    alongside the gate's verdict, so the test can check the gate was right rather than merely that
    it fired. A gate that admitted everything would pass the first half of this function.
    """
    assert isinstance(activation, dict)
    assert activation["activated"] is True
    assert activation["audit_ok"] is True
    assert int(activation["audit_entries"]) >= 3
    refused = activation["refused"]
    assert isinstance(refused, dict)
    assert "always_opus" in refused
    assert refused["always_opus"] == "NotEligibleError"

    verdicts = activation["verdicts"]
    assert isinstance(verdicts, dict)
    incumbent_truth = float(activation["incumbent_true_value"])
    admitted = {name: v for name, v in verdicts.items() if v["eligible"]}
    assert admitted, "the demo is meant to show a promotion, not only refusals"
    for name, verdict in admitted.items():
        assert float(verdict["true_value"]) > incumbent_truth, f"{name} was admitted but is worse"
    assert verdicts["always_opus"]["eligible"] is False
    assert float(verdicts["always_opus"]["true_value"]) <= incumbent_truth


@pytest.mark.usefixtures("small_demo")
def test_the_demo_runs_end_to_end_and_reports_what_the_readme_claims(tmp_path: Path) -> None:
    summary = demo.run_demo(tmp_path)

    # Reproducibility: the workload and environment are identified by content, not by a timestamp.
    assert len(summary["workload_digest"]) == 64
    assert len(summary["environment_digest"]) == 64

    # Learning: cheaper than always-Sonnet at comparable quality, with sublinear in-class regret.
    learning = summary["learning"]
    assert isinstance(learning, dict)
    assert float(learning["linucb_saving_vs_sonnet"]) > 0.30
    assert abs(float(learning["linucb_quality_delta_vs_sonnet"])) < 0.01
    assert float(learning["linucb_in_class_regret"]) > 0.0

    # Off-policy evaluation: DR is the most accurate of the three against the exact values.
    ope = summary["ope"]
    assert isinstance(ope, dict)
    assert set(ope) == {"ips", "snips", "dr"}
    assert float(ope["dr"]["rmse"]) <= float(ope["ips"]["rmse"])
    for method in ("ips", "snips", "dr"):
        assert 0.0 <= float(ope[method]["coverage_95"]) <= 1.0
        assert float(ope[method]["mean_ess"]) > 0.0

    # Adaptation: the shift really bites, and the discounted posterior recovers sooner.
    adaptation = summary["adaptation"]
    assert isinstance(adaptation, dict)
    assert set(adaptation) == {"linucb_stationary", "linucb_discounted"}
    stationary = adaptation["linucb_stationary"]
    discounted = adaptation["linucb_discounted"]
    assert isinstance(stationary, int), "the stationary policy should have recovered eventually"
    assert isinstance(discounted, int)
    assert stationary > 0, "a shift that never dents the ratio would prove nothing"
    assert discounted <= stationary

    check_governance(summary["activation"])

    # Poisoning: undefended feedback costs real policy value; the shipped pipeline nearly erases it.
    poisoning = summary["poisoning"]
    assert isinstance(poisoning, list)
    by_defense = {row["defenses"]: row for row in poisoning}
    assert set(by_defense) == {"n/a", "none", "clipping only", "full"}
    assert float(by_defense["none"]["value_drop"]) > float(by_defense["full"]["value_drop"])
    assert float(by_defense["full"]["value_drop"]) < 0.02

    # The live path: requests served through RouterService, one tenant hitting its cap.
    service = summary["service"]
    assert isinstance(service, dict)
    assert int(service["requests_served"]) > 50
    assert int(service["requests_denied_by_budget"]) > 0
    assert int(service["decisions_logged"]) == int(service["requests_served"])
    assert int(service["policy_updates"]) > 0
    assert set(service["spend_by_tenant_usd"]) == {"tenant_main", "tenant_small"}

    assert len(str(summary["mlflow_run_id"])) == 32
    assert float(summary["seconds"]) > 0.0


@pytest.mark.usefixtures("small_demo")
def test_the_demo_is_reproducible_and_cleans_its_workdir(tmp_path: Path) -> None:
    """Two runs print the same numbers, and the second does not inherit the first one's state."""
    first = demo.run_demo(tmp_path)
    marker = tmp_path / "left-behind.txt"
    marker.write_text("stale", encoding="utf-8")
    second = demo.run_demo(tmp_path)
    assert not marker.exists()
    assert first["workload_digest"] == second["workload_digest"]
    assert first["learning"] == second["learning"]
    assert first["ope"] == second["ope"]
    assert first["poisoning"] == second["poisoning"]


@pytest.mark.usefixtures("restore_logging", "small_demo")
def test_the_cli_demo_prints_a_report_then_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    # The tables come first, then exactly one JSON object.
    assert "## 1. Learning" in output
    start = output.index("{", output.index("## 7."))
    payload = json.loads(output[start:])
    assert payload["demo"] == "bandit-llm-router"
    assert len(payload["workload_digest"]) == 64
    # Service logs go to stderr, so stdout stays a clean report.
    assert "route_decision" not in output


def test_the_cli_generates_a_key_and_prints_only_its_digest_for_storage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["generate-api-key", "--scope", "approve"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["scope"] == "approve"
    assert payload["sha256"] == hash_api_key(payload["api_key"])
    # The instruction names the right environment variable and never repeats the raw key.
    assert "APP_APPROVER_API_KEY_SHA256" in captured.err
    assert payload["api_key"] not in captured.err


@pytest.mark.parametrize("scope", ["route", "feedback", "admin", "approve"])
def test_every_scope_has_its_own_environment_variable(
    scope: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["generate-api-key", "--scope", scope]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["scope"] == scope
    assert scope.upper().replace("APPROVE", "APPROVER") in captured.err


def test_the_cli_verifies_the_activation_audit_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "activations.audit.jsonl"
    log = AuditLog(path)
    log.append("operator_a", "policy_activated", {"candidate_version": "linucb-abc"})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_the_cli_serves_uvicorn_with_the_hardened_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_the_module_entry_point_works(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bandit_llm_router", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("bandit_llm_router", run_name="__main__")
    assert exit_info.value.code == 0
    assert "sha256" in capsys.readouterr().out
