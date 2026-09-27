"""Micro-benchmarks for the baseline components, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py           # run, write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if a threshold is exceeded

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts. The project's own
measurements -- learning, off-policy evaluation, adaptation, robustness and routing overhead -- are
in bench_regret.py, bench_ope.py and bench_router.py.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

# bench/ is a script directory on sys.path, not an installed package.
from _common import machine, threshold_problems, write_results
from bandit_llm_router.agentic.audit import AuditLog, verify_file
from bandit_llm_router.agentic.providers.base import ChatMessage, LLMRequest
from bandit_llm_router.core.ratelimit import RateLimiter
from bandit_llm_router.core.redaction import redact_text
from bandit_llm_router.ml.drift import DriftMonitor


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def bench_request_hash() -> dict[str, float]:
    messages = tuple(ChatMessage.user(f"message {i} " + "x" * 400) for i in range(10))
    request = LLMRequest(system="bench", messages=messages)
    return _timed(request.sha256, runs=2_000)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return _timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=200
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(1_000):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return _timed(lambda: verify_file(path), runs=10, warmup=1)


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def run() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "llm_request_sha256_10_messages": bench_request_hash(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_1000_entries": bench_audit_verify(workdir),
            "drift_check_10k_vs_10k": bench_drift(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="baseline micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    write_results("baseline.json", {"machine": machine(), "results": results})
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    if args.check:
        problems = threshold_problems("bench_baseline", results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
