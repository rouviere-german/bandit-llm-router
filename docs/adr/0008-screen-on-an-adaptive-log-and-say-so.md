# 8. Screen on the live learner's own log, with a block bootstrap and a step window

- Status: accepted
- Date: 2026-09-28

## Context

`bench/bench_ope.py` measures the DR interval's coverage with a deliberately **frozen** logging
policy, and `src/bandit_llm_router/experiments.py` explains why: an online learner's log is not
i.i.d. in its rows, so a bootstrap over those rows would not be measuring what it claims to.

The shipped gate does not get that luxury. `RouterService.screen` evaluates candidates on
`store.fetch_logged_rows(...)` -- the decision log the **live** policy wrote, and that policy updates
on the first accepted reward report for every decision. Each row's propensity therefore depends on the
rewards of the rows before it. The published 0.983-1.000 coverage was measured in one regime and
presented, in README and MODEL_CARD, as characterising the gate's intervals in another.

Filtering cannot rescue it. `policy_version` is a hash of *configuration* -- arms, dim, ridge,
discount, propensity floor, feature version, alpha -- so every row a learning LinUCB ever writes
carries the same version no matter how far its posteriors have travelled.

## Decision

Three things, in this order of importance.

1. **Say it.** README Results has a subsection stating that the headline coverage is for a frozen
   logging policy and does not transfer unchanged to the gate's own log; MODEL_CARD carries it as a
   second calibration caveat; README Non-goals states that valid inference on adaptively collected
   data is not solved here.
2. **Measure it.** `experiments.run_adaptive_coverage` logs the *same* candidates and seeds twice --
   once frozen, once with a cold-started learner still updating -- and scores the DR interval under
   both the row bootstrap and a block bootstrap, sharing one logging run per seed so the only
   difference is the resampling scheme. `bench/bench_ope.py` publishes the four cells, plus an
   arm-share drift figure (total variation between the first and last tenth of each log's arm shares)
   so the comparison cannot pass vacuously on a log that never moved.
3. **Mitigate it.** `ope/estimators.py` takes an optional `block_size` and then resamples contiguous
   runs of rows -- the circular block bootstrap of Politis and Romano (1992) -- and
   `ScreeningConfig.bootstrap_block_size` is part of the hashed report, so the choice is auditable.
   The service sets it to 250 by default (`APP_SCREENING_BOOTSTRAP_BLOCK_SIZE`).
   `fetch_logged_rows` and `POST /v1/policies/{digest}/screen` take `min_policy_step` /
   `max_policy_step`, because `policy_step` is the only column that distinguishes a posterior that has
   moved from one that has not.

## Consequences

- **The measured degradation on this workload is zero**: DR coverage 0.967 for the frozen logger and
  0.967 for the learner, under both bootstraps, at 8,000 rows over 60 trials each, with drift 0.023
  against 0.069. Reporting that is the point of the exercise. The reason the estimator survives is
  worth stating: importance weighting is *conditionally* unbiased, so each row's term is a martingale
  difference sequence around the target and the naive variance estimate stays roughly right even
  though the rows are dependent.
- **The block bootstrap is therefore insurance, not a fix for a measured defect.** It is on by default
  anyway: one simulator is not a proof, and the failure it guards against (dependence that moves the
  *mean* of the per-row terms, e.g. a drifting environment rather than a drifting policy) is real and
  would be invisible to a row bootstrap. `tests/unit/test_ope.py::test_the_row_bootstrap_understates_the_interval_on_a_serially_dependent_log`
  constructs exactly that case and shows the interval widening by more than 4x.
- **The adaptive section uses its own, shorter log (8,000 rows) and does not follow `--logs`.** A
  LinUCB here converges within a few thousand requests, so in a 40,000-row window most rows come from
  a policy that has stopped moving, which dilutes the effect being measured. Measured both ways: at
  40,000 rows the drift was 0.035 against a 0.023 floor and coverage was 1.000 in all four cells.
- **What is still not solved.** None of this is a validity proof for adaptively collected data.
  Anytime-valid confidence sequences, or a martingale concentration bound in place of the bootstrap,
  are the real answer. Recorded as a README non-goal rather than implemented.
- Alternatives rejected:
  - *screen only on rows from a frozen block of the log* -- clean in theory, but a learning policy's
    "frozen block" is the interval between two updates, which is a handful of rows; the `policy_step`
    window is the usable version of this idea and is exposed for an operator who wants it;
  - *freeze the live posterior for the duration of a screening window* -- stops the service learning
    while a governance decision is pending, which is a real availability cost for a statistical
    improvement we cannot measure;
  - *say nothing, since the measured effect is zero* -- the number was produced in a different regime
    than the one it was quoted about, and that gap is what needed disclosing, independently of how
    large it turned out to be.
