"""MLflow run logging on the local file backend, and the Markdown tables the benches print.

Tracking is the index, not the source of truth: the authoritative reproducibility record is the run
manifest plus the content-addressed registry. What matters here is that logging a run needs no
server and no network, so the demo and the tests work offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bandit_llm_router.reporting import format_number, markdown_table
from bandit_llm_router.tracking import EXPERIMENT_NAME, list_runs, log_policy_run, tracking_uri

pytestmark = pytest.mark.unit


# --- tracking -----------------------------------------------------------------------------------


def test_a_run_is_logged_and_read_back_locally(tmp_path: Path) -> None:
    directory = tmp_path / "mlruns"
    run_id = log_policy_run(
        directory,
        run_name="linucb-seed-7",
        params={"policy_version": "linucb-abc123", "seed": 7, "workload_digest": "d" * 64},
        metrics={"mean_reward": 0.61, "final_regret": 120.5},
        tags={"kind": "simulation"},
    )
    assert len(run_id) == 32
    runs = list_runs(directory)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["run_name"] == "linucb-seed-7"
    assert runs[0]["params"]["seed"] == "7"
    assert runs[0]["metrics"]["mean_reward"] == pytest.approx(0.61)


def test_the_reproducibility_fields_every_run_must_carry(tmp_path: Path) -> None:
    """Seed, workload digest and policy version, so a number can be traced back to its inputs."""
    directory = tmp_path / "mlruns"
    log_policy_run(
        directory,
        run_name="run",
        params={"seed": 1, "workload_digest": "a" * 64, "policy_version": "linucb-1"},
        metrics={"mean_reward": 0.5},
    )
    params = list_runs(directory)[0]["params"]
    assert {"seed", "workload_digest", "policy_version"} <= set(params)


def test_artifacts_ride_along_with_the_run(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"seed": 7}', encoding="utf-8")
    run_id = log_policy_run(
        tmp_path / "mlruns",
        run_name="with-artifact",
        params={"seed": 7},
        metrics={"value": 1.0},
        artifacts=[manifest],
    )
    assert run_id


def test_an_over_long_parameter_is_truncated_not_refused(tmp_path: Path) -> None:
    directory = tmp_path / "mlruns"
    log_policy_run(
        directory, run_name="long", params={"notes": "x" * 5_000}, metrics={"value": 1.0}
    )
    assert len(list_runs(directory)[0]["params"]["notes"]) <= 500


def test_an_empty_directory_lists_nothing(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "never-used") == []


def test_the_tracking_uri_prepares_the_layout_the_file_store_expects(tmp_path: Path) -> None:
    directory = tmp_path / "mlruns"
    uri = tracking_uri(directory)
    assert uri.startswith("file:")
    assert (directory / ".trash").is_dir()
    assert EXPERIMENT_NAME == "bandit-llm-router"


def test_a_recreated_directory_still_works(tmp_path: Path) -> None:
    """The demo wipes its working directory between runs; MLflow caches its store per URI."""
    directory = tmp_path / "mlruns"
    log_policy_run(directory, run_name="first", params={"seed": 1}, metrics={"value": 1.0})
    import shutil

    shutil.rmtree(directory)
    log_policy_run(directory, run_name="second", params={"seed": 2}, metrics={"value": 2.0})
    assert [run["run_name"] for run in list_runs(directory)] == ["second"]


# --- reporting ----------------------------------------------------------------------------------


def test_a_table_renders_uniform_rows() -> None:
    rows = [
        {"policy": "linucb", "mean_reward": 0.6123456, "eligible": True},
        {"policy": "always_opus", "mean_reward": 0.5, "eligible": False},
    ]
    table = markdown_table(rows)
    lines = table.splitlines()
    assert lines[0] == "| policy | mean_reward | eligible |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| linucb | 0.6123 | yes |"
    assert lines[3] == "| always_opus | 0.5 | no |"


def test_an_empty_table_says_so() -> None:
    assert markdown_table([]) == "(no rows)"


def test_a_missing_key_renders_as_an_empty_cell() -> None:
    table = markdown_table([{"a": 1, "b": 2}, {"a": 3}])
    assert table.splitlines()[-1] == "| 3 |  |"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (True, "yes"),
        (False, "no"),
        (0.5, "0.5"),
        (0.6123456, "0.6123"),
        (1.0, "1"),
        (0.0, "0"),
        (12_345_678.9, "12345678.9"),
        (7, "7"),
        ("text", "text"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
    ],
)
def test_numbers_are_formatted_in_exactly_one_place(value: object, rendered: str) -> None:
    assert format_number(value) == rendered
