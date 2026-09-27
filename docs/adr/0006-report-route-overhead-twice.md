# 6. Report route overhead twice: with and without the shared budget file

- Status: accepted
- Date: 2026-09-27

## Context

`bench/bench_router.py` measures per-request latency on the live routing path with the offline
provider: featurize, decide, restrict, reserve, call, commit, log to DuckDB. A single p95 for that path
is easy to quote and hard to interp, because one component dominates it for a reason that has nothing to
do with routing.

The per-tenant budget guard keeps each tenant's spend in a JSON file under a file lock, and reloads it
before every change, so several workers share one cap instead of each spending the full amount. That
costs two atomic writes plus a reload per reserve and per commit -- disk work, measured on whatever disk
the bench happens to run on. Quoting one number would let a reader conclude that *routing* costs
hundreds of microseconds, when the decision itself is a few matrix-vector products.

## Decision

The bench measures the same path twice and the README reports both rows:

- **shared budget file** -- the default deployment: lock, reload, write, on every reserve and commit.
- **in-memory budget guard** -- the same code path with the cap held in process.

The difference is the price of the shared cap; the second row is the cost of the decision work
(featurize, decide, restrict, call the offline provider, log to DuckDB). Both rows carry their own
threshold in `bench/thresholds.json`, and both assert `lost_decisions == 0` so a faster configuration
cannot pass by dropping rows from the decision log.

## Consequences

- The README can say which half is which, and a reader deciding whether to run one worker with an
  in-process cap or several workers with a shared one has the number that decides it.
- Two thresholds instead of one, and a regression in the shared-file path does not mask an improvement
  in the decision path or vice versa.
- The in-memory row is a *measurement configuration*, not a recommended deployment: with it, `n` workers
  each spend the full cap. Anyone tempted by the faster number needs a shared store (the file, or a real
  one), which the README's Non-goals states we do not provide beyond a local file lock.
- The absolute numbers are single-machine and disk-bound in the shared case. They are for spotting
  regressions between runs on the same machine, which is why the thresholds are several times the
  measured values: CI runners are noisy, and the point is to catch an order-of-magnitude change or a
  broken invariant, not a few percent.
