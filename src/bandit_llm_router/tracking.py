"""Policy versioning in MLflow, on the local file backend (mlflow-skinny, no server).

Each simulation or screening run logs the things that decide whether a number can be trusted again
later: the workload digest and simulator seed, the environment digest, the policy version and its
hyperparameters, the propensity floor, the reward trade-off, and the headline metrics. The run
manifest (seed, package versions, dataset fingerprints) goes along as an artifact.

MLflow is the index, not the source of truth. The authoritative reproducibility record is the run
manifest plus the content-addressed model registry; MLflow makes runs comparable. That matters
because the file store is in upstream maintenance mode -- noted in the README's non-goals -- and
because nothing here should require a tracking server to be reproducible. Every call is local: no
server, no database, nothing uploaded, so the offline demo and the tests work with no network.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# Set before mlflow is imported anywhere: skip the agent hint, no system-metrics thread, and opt in
# to the filesystem store that MLflow >= 3.4 otherwise refuses.
for name, value in (
    ("MLFLOW_DISABLE_AGENT_HINT", "1"),
    ("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false"),
    ("MLFLOW_ALLOW_FILE_STORE", "true"),
):
    os.environ.setdefault(name, value)

__all__ = ["EXPERIMENT_NAME", "list_runs", "log_policy_run", "tracking_uri"]

EXPERIMENT_NAME = "bandit-llm-router"
_MAX_PARAM_CHARS = 500


def tracking_uri(directory: Path) -> str:
    """File URI for ``directory``, with the layout the file store expects already created.

    MLflow caches one store per tracking URI for the life of the process, and the cached store
    assumes its folders still exist. The demo recreates its working directory between runs, so the
    ``.trash`` folder is created here rather than relying on MLflow to recreate it.
    """
    (directory / ".trash").mkdir(parents=True, exist_ok=True)
    return directory.resolve().as_uri()


def log_policy_run(
    directory: Path,
    *,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: Iterable[Path] = (),
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log one run to the local file backend and return its run id."""
    import mlflow  # noqa: PLC0415 - a slow import; only pay for it when tracking

    mlflow.set_tracking_uri(tracking_uri(directory))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        mlflow.log_params({key: str(value)[:_MAX_PARAM_CHARS] for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)


def list_runs(directory: Path) -> list[dict[str, Any]]:
    """Runs in the experiment as plain dicts (run id, name, params, metrics), newest first."""
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(tracking_uri(directory))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
        for run in runs
    ]
