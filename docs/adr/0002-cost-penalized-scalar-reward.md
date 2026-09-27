# 2. Optimize a cost-penalized scalar reward, not a constrained objective

- Status: accepted
- Date: 2026-09-27

## Context

Routing is a two-objective decision: answer quality and money. The natural statement of the goal is
constrained -- "maximize expected quality subject to spend per thousand requests below `B`" -- and
there is a large literature on constrained and budgeted bandits that solves exactly that, typically
with a Lagrangian whose multiplier is learned online (Badanidiyuru, Kleinberg and Slivkins,
*Bandits with Knapsacks*, FOCS 2013; Agrawal and Devanur, *Bandits with concave rewards and convex
knapsacks*, EC 2014).

The machinery in this repo, however, assumes a single bounded scalar reward in several places at
once:

- the per-arm ridge posteriors treat the reward as the regression target, and their noise scale is
  only meaningful if the target is bounded;
- IPS, SNIPS and DR estimate the value of *one* scalar; a vector-valued or constrained value has no
  single "policy value" to compare against an incumbent;
- the screening gate compares one number, the DR lower bound, against one incumbent number.

So the choice of objective is not a modelling detail we can defer. It decides whether off-policy
evaluation and the activation gate mean anything.

## Decision

The objective is a fixed scalar with the cost folded in:

```text
reward = clip(quality - cost_weight * min(1, cost / cost_reference), 0, 1)
```

with `quality` in `[0, 1]`, `cost` the call's estimated spend in micro-USD, and `cost_weight` and
`cost_reference_micros` frozen in `RewardConfig` (`src/bandit_llm_router/reward.py`). Three
properties are deliberate:

- **Bounded.** The inner `min(1, ...)` caps the penalty at `cost_weight` and the outer clip keeps the
  reward in `[0, 1]`, so one freak expensive call cannot dominate a posterior or blow up an
  importance weight.
- **Versioned, not tuned live.** `RewardConfig` is frozen and contributes to the policy version
  digest, so a policy trained under one trade-off is never screened against logs shaped under
  another. `cost_weight` is the exchange rate between quality and dollars; changing it changes what
  "better" means, which is a governance event, not a hyperparameter sweep.
- **Priced from a reviewed file.** Token prices live in `config/pricing.toml`, whose models must be
  on the allowlist in `src/bandit_llm_router/arms.py`, and `tests/unit/test_arms.py` asserts they
  agree with the prices that charge the LLM budget ledger. Reward and spend cannot disagree.

Hard spend limits are kept as a *separate* mechanism rather than folded into the objective: the
per-tenant guard in `src/bandit_llm_router/budgets.py` refuses any arm whose worst-case cost does not
fit the remaining cap, before the provider is called.

## Consequences

- OPE, screening and the incumbent comparison all work on one comparable number, which is what makes
  the activation gate expressible at all.
- A hard budget is enforced by an admission check with a proved invariant (`spent <= cap`, property
  tested in `tests/unit/test_budgets.py`) instead of by a soft penalty that a learner could decide to
  pay. A Lagrangian respects its constraint only in expectation and only after the multiplier
  converges; a tenant cap has to hold on every request.
- The reported "cost reduction at equal quality" is a claim about *this* `cost_weight`. The README
  reports mean quality next to the saving precisely because a single reward number could hide a
  quality regression, and `bench/bench_regret.py` enforces a threshold on the quality shortfall as
  well as on the saving.
- We give up the ability to say "hit exactly this monthly spend": the router minimizes a penalized
  objective and the cap truncates it. Choosing `cost_weight` to land on a target spend is a manual
  calibration loop, which is recorded as a non-goal in the README.
- `quality` offline is a simulator proxy, so every number the benches produce validates the learning
  and evaluation machinery, not a real quality/price frontier. MODEL_CARD.md states this.
