"""Shared plumbing for the bench scripts: results files, machine facts and threshold checking.

Every bench writes one JSON file under ``bench/results/`` and, with ``--check``, compares its own
section of ``bench/thresholds.json``. Thresholds are one-sided upper bounds, so a metric that must
stay *high* is expressed as the shortfall below its target -- one comparison direction keeps the
CI check trivial to read and impossible to get backwards.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

__all__ = ["ROOT", "machine", "threshold_problems", "write_results"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def machine() -> dict[str, object]:
    """The facts a reader needs to judge a timing, recorded next to every result."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def write_results(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def threshold_problems(section: str, results: dict[str, dict[str, float]]) -> list[str]:
    """Metrics in ``section`` over their upper bound, plus any benchmark that vanished."""
    document: dict[str, Any] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    if section not in document:
        return [f"thresholds.json has no section {section!r}"]
    problems: list[str] = []
    for name, limits in document[section].items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            if metric not in results[name]:
                problems.append(f"{name}.{metric}: metric missing")
                continue
            value = float(results[name][metric])
            if value > float(limit):
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems
