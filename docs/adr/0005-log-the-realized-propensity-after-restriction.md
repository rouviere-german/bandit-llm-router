# 5. Log the propensity of the distribution that actually ran

- Status: accepted
- Date: 2026-09-27

## Context

The policy proposes a distribution over arms, but three things can override it between the proposal
and the call:

1. an arm's circuit breaker is open (the provider has been failing);
2. an arm's worst-case cost does not fit the tenant's remaining budget;
3. the call to the chosen arm fails, and the request is redirected once to a healthy arm.

Each of these means the action was drawn from something other than what the policy proposed. Off-policy
evaluation does not care what the policy *wanted*; it needs the probability of the action under the
mechanism that produced it. If the router logs the policy's own distribution while a breaker was
silently rerouting half the traffic, every importance weight in the log is wrong -- and wrong in a way
no amount of data fixes, because the log itself is mislabelled.

This is the failure mode that makes production OPE untrustworthy in practice, so it needs an explicit
decision rather than an implementation detail.

## Decision

The log always carries the propensity of the distribution that ran.

- **Pre-call restriction.** Blocked arms (breaker open, or cost ceiling over the remaining cap) are set
  to exactly zero and their mass is moved to the first available arm in the catalog's
  `fallback_priority` order (`restrict_distribution` in `src/bandit_llm_router/policies/base.py`). The
  restricted distribution is a genuine probability distribution, the action is drawn from it, and its
  probability for the drawn action is what is written to the decision log. When nothing is available
  the request is refused rather than served off-distribution.
- **Post-call redirection.** When the chosen arm's call fails, the breaker trips, the distribution is
  restricted again without that arm, and the request is redirected once. The row is flagged
  (`served_by_fallback`, `breaker_adjusted`) and carries the redirected propensity.
- **The flags are logged, not just the number**, so an analysis can exclude redirected rows entirely if
  it wants a clean sample instead of a corrected one.

## Consequences

- Importance weights are computed against the real mechanism, which is what makes the OPE accuracy
  bench meaningful: it recovers the simulator's exact policy values under restriction and floors, not
  just in the unrestricted case.
- Restriction is *also* how the budget cap stays compatible with exploration. An arm the policy would
  like to try is simply unavailable when its ceiling does not fit, its mass moves to an affordable arm,
  and the realized propensity is logged. Exploration therefore cannot push a tenant over its cap, and
  the OPE screening gate can still use the log.
- **The honest caveat.** Redirecting after a failure treats a provider failure as a property of the arm
  at that moment, independent of the request. If an arm failed preferentially on particular contexts --
  long prompts, say -- then the post-failure propensity is only approximately right, because the
  redirection probability would depend on `x`. We accept the approximation and flag the rows instead of:
  - *modelling the failure probability per context and folding it into the propensity*, which makes the
    logged propensity depend on a fitted model of an outage, i.e. replaces a small known bias with an
    unknown one;
  - *dropping redirected requests from the log*, which loses exactly the rows where cost and quality are
    most unusual and biases the remaining sample;
  - *failing the request instead of redirecting*, which trades a small statistical bias for a real
    availability loss -- the wrong trade for a router whose job is to serve the request.
- Propensity floors interact with all of this: the floor is applied by the policy before restriction, so
  a floor guarantees a minimum probability only among arms that were actually available. The minimum
  *realized* propensity is exported as a metric (`propensity_min`) so a collapse in support is visible
  rather than inferred later from unstable estimates.
