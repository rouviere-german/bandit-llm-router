# 4. Handle non-stationarity with one discounted, re-regularized Gram matrix

- Status: accepted
- Date: 2026-09-27

## Context

Model quality is not stationary. A provider ships an update and one arm gets better at code and worse
at extraction; a rubric is rewritten and every reward shifts. A textbook per-arm ridge posterior
treats all history as equally valid, so after such a shift it holds a confident, wrong belief: the
confidence width has shrunk to nothing exactly where the evidence has gone stale, and the policy stops
exploring the arm it needs to re-measure. In simulation that shows up as a router that never recovers
its pre-shift reward ratio.

The repo needs *some* forgetting mechanism, and the choice has to survive review, because the shift
experiment is one of the evaluation metrics the README reports.

## Decision

Each arm keeps one Gram matrix and one right-hand side, updated with a discount and a
re-regularization term:

```text
A_k  <-  discount * A_k + (1 - discount) * ridge * I + w * x x^T
b_k  <-  discount * b_k + w * r * x
```

(`src/bandit_llm_router/policies/linear.py`). With `discount = 1` this is exactly the LinUCB posterior
of Li et al. (*A Contextual-Bandit Approach to Personalized News Article Recommendation*, WWW 2010).
With `discount < 1`, evidence decays exponentially with a half-life of `ln(2) / -ln(discount)` updates,
so uncertainty grows again in directions that stop being refreshed and exploration restarts on its
own.

The `(1 - discount) * ridge * I` term is the part that is easy to omit and costly to omit. Without it,
discounting shrinks `A_k` toward zero in directions that stop being explored, the inverse blows up,
and confidence widths diverge instead of merely widening. With it, `A_k` decays toward `ridge * I` --
the prior -- which is the honest limit: no recent evidence means prior uncertainty, not infinite
uncertainty.

This is a deliberately simplified form of the weighted linear bandit of Russac, Vernade and Cappe
(*Weighted Linear Bandits for Non-Stationary Environments*, NeurIPS 2019). Their estimator tracks a
**second** matrix accumulating squared weights, and uses it to build a confidence ellipsoid that is
correctly calibrated for the weighted least-squares estimator; that is what their regret bound needs.

### Alternatives rejected

- **The full two-matrix weighted linear bandit.** Correct confidence widths and a regret bound, at the
  cost of a second `d x d` matrix per arm, a second inverse, and a width formula with two more
  hyperparameters to justify. We do not claim a regret bound: the README claims a measured sublinear
  fitted log-log slope, and the shift bench measures recovery directly. Paying the complexity for a
  guarantee we do not assert, in a repo whose point is the routing and OPE machinery, was not worth it.
  If the widths turn out to be miscalibrated the symptom is visible in the bench (recovery too slow,
  or exploration that never settles), which is why that bench has thresholds.
- **Sliding window.** Conceptually simpler, but it means storing the last `W` observations per arm and
  rebuilding or downdating the Gram matrix, which costs memory proportional to the window and gives a
  hard boundary where a single old observation drops out and the posterior jumps. Exponential decay is
  `O(d^2)` state and moves smoothly.
- **Change-point detection with a posterior reset.** Sharper in principle for an abrupt shift, but a
  false positive throws away all the evidence, and tuning the detector is a second learning problem
  whose failures are hard to attribute. Discounting degrades gracefully instead.
- **Just periodic retraining.** Does nothing between retrains, which is where the loss is.

## Consequences

- One tunable, `discount`, with a physical reading (an evidence half-life). It is part of the policy
  version, so a discounted candidate and a stationary incumbent are distinguishable in the audit.
- `tests/unit/test_policies.py` checks that the maintained inverse agrees with a fresh solve, that a
  discounted posterior forgets at the expected rate, and that widths grow rather than diverge when an
  arm stops being played.
- `bench/bench_router.py` reports requests-to-recover for stationary and discounted posteriors after a
  scheduled shift, with a threshold that fails CI if the discounted variant stops beating the
  stationary one. That is the measurement standing in for the regret bound we did not import.
- With `discount == 1` an update is a rank-one change, so the inverse is maintained by
  Sherman-Morrison in `O(d^2)`; the full-matrix decay path recomputes it. Rank-one updates accumulate
  floating-point error, so the inverse is recomputed exactly every `refresh_every` updates.
- We cannot state a non-stationary regret bound for this variant, only measured behaviour on a
  simulator whose shift we wrote. The README's Non-goals says so.
