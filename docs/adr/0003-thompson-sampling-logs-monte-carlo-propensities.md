# 3. Thompson sampling is deployed as its Monte-Carlo action distribution

- Status: accepted
- Date: 2026-09-27

## Context

Every off-policy estimator in this repo divides by the probability with which the logging policy took
the action it took. For LinUCB and epsilon-greedy that probability is available in closed form: put
the mass on the argmax of a deterministic score, mix with the propensity floor, done.

Thompson sampling has no such closed form. A step draws `theta_k` from each arm's posterior and plays
the argmax of `theta_k . x`; the probability that arm `k` wins is the probability that a particular
Gaussian is the maximum of `K` Gaussians, which has no elementary expression for `K > 2`. Yet the
logged number is not optional: an importance weight built from a wrong propensity is not noisy, it is
biased, and it stays biased however much data is collected. Getting this wrong would quietly
invalidate every estimate, every screening decision and every number in the README's Results
section.

## Decision

The deployed Thompson policy *is* the Monte-Carlo distribution, not a sampled argmax:

1. Take `mc_samples` posterior draws per arm for the current context.
2. Let `p_k` be the fraction of draws won by arm `k`.
3. Apply the propensity floor to `p`.
4. Sample the action from `p`, and log `p[action]`.

The logged propensity is then *exactly* the probability with which that action was taken, conditional
on the draws. That is all importance weighting needs, because the draws are independent of the reward:
conditioning on them leaves the estimator unbiased. Implementation in
`src/bandit_llm_router/policies/linear_bandit.py`.

For one context the posterior of `theta_k . x` is `N(theta_k . x, sigma^2 * x^T A_k^-1 x)` and the
arms are independent, so a draw costs one standard normal per arm, not a `d`-dimensional multivariate
sample. `mc_samples` draws therefore cost `O(K * mc_samples)` after the `O(d^2)` quadratic forms that
LinUCB needs anyway.

### Alternatives rejected

- **Log the realized argmax with propensity 1.** Simplest, and wrong: it claims a deterministic
  policy that was not run. Importance weights become 1, IPS collapses to the logged average reward,
  and no counterfactual candidate can be evaluated at all.
- **Log the realized argmax with the epsilon-greedy floor as its propensity.** Still wrong, and worse
  because it looks defensible: the floor is a lower bound on the action probability, not the action
  probability, so weights are systematically too large for the arm the posterior favours.
- **Numerically integrate the max-of-Gaussians probability.** Exact to quadrature error, but it costs
  a `K`-dimensional integral per request on the serving path for a quantity we already obtain to
  `O(1/sqrt(mc_samples))` accuracy by sampling -- and the sampled version has the stronger property
  of being the *actual* mechanism rather than a model of it.
- **Keep sampling but reconstruct propensities offline at evaluation time.** Requires storing the
  posterior state per request, or recomputing it by replaying the update stream, and any drift
  between the reconstruction and what the server really did reintroduces the bias we were avoiding.

## Consequences

- Thompson sampling's logged propensities are correct by construction, and the OPE accuracy bench
  (`bench/bench_ope.py`) measures bias and CI coverage against the simulator's exact policy values as
  a check on exactly this.
- The deployed policy is a *smoothed* Thompson sampler. With finite `mc_samples` its action
  distribution differs slightly from the idealized one, most visibly in that an arm which wins a
  small fraction of draws keeps a small but non-zero probability. That is a change in the policy, not
  an error in the logging, and the policy that was screened is the policy that runs.
- `mc_samples` is part of the policy version, so a candidate screened at one sampling resolution
  cannot be activated at another.
- Propensities are bounded below by the floor and above by 1, and
  `tests/unit/test_policies.py` property-tests that the distribution sums to 1 and respects the floor
  for arbitrary contexts and posterior states.
