# 7. Price the screening cost gate against reachable mass, not reachability

- Status: accepted
- Date: 2026-09-28
- Supersedes the original formulation of gate 3 in `src/bandit_llm_router/ope/screening.py`.

## Context

Gate 3 of the promotion screen is supposed to refuse a candidate whose exploration would bust a
budget over a horizon of requests. Its first implementation computed, for each logged context, the
price of the most expensive arm the candidate could *reach*:

```python
reachable = probabilities > 0.0
worst = np.max(np.where(reachable, ceiling_cost_micros, -np.inf), axis=1).mean()
```

That is a genuine supremum on spend, and it is useless as a gate here, for one reason: **every policy
in this repo applies a propensity floor.** A floor gives every arm at least `floor` probability, so
every arm is reachable for every candidate, and the maximum collapses to the dearest arm's ceiling
regardless of what the candidate would actually do. Measured: a trained LinUCB that sends 82% of its
traffic to Haiku and a static always-Opus policy were priced *identically*, and the demo printed the
same 668 USD for all three of its candidates.

A gate that returns one number for every candidate rejects all of them or none of them. It was
documented in README and CLAIMS as a per-candidate check, which it was not.

## Decision

Price the horizon against the candidate's own action distribution, and keep the supremum as a
separate, clearly labelled field.

For each logged row `i`, take the candidate's distribution `p_i` and the arms' **ceiling** prices
`c_i` (the full `max_output_tokens`, not the expected output length):

- `mu_i = sum_a p_ia * c_ia` -- mean spend on that context at ceiling prices;
- `var_i = sum_a p_ia * c_ia^2 - mu_i^2` -- the arm lottery's variance inside one request.

Over a horizon `H` of requests drawn from the same context distribution as the log, total spend at
ceiling prices has mean `H * E[mu]` and variance `H * (E[var] + Var[mu])`: the lottery inside a
request plus the spread of the contexts themselves. The gate compares

```
H * E[mu] + z * sqrt(H * (E[var] + Var[mu]))
```

against `max_exploration_usd`, with `z = cost_bound_z` (default 3). The supremum -- the dearest
reachable arm on every request -- is still computed and reported as
`CostBound.dearest_arm_usd_per_request`, because it is the right number to look at when sizing an arm
set, and it is *not* the gate.

## Consequences

- **The gate discriminates.** In the demo, at a horizon of 100,000 requests: a trained cost-aware
  policy is priced at 225 USD, `always_sonnet` at 274 and `always_opus` at 654, against a 300 USD cap.
  So `always_opus` is refused on cost and `always_sonnet` is not, which is the difference that makes
  it a gate. `tests/unit/test_screening.py::test_the_cost_bound_separates_candidates_by_how_much_dear_mass_they_carry`
  fails against the old implementation.
- **The floor's forced exploration is still charged.** A candidate with zero mass on the dearest arm
  prices strictly below one that carries only the floor's 2%, which is the behaviour a propensity
  floor's cost should have.
- **It is a high-probability bound, not a supremum, and the docs now say so.** At `z = 3` the
  normal-approximation tail is about 1e-3. A pathological context stream could exceed it. We accept
  that because the *hard* bound on spend does not live here at all: `TenantBudgetGuard` reserves each
  call's cost ceiling against the tenant's remaining cap **before** the provider is called, so
  `spent <= cap` is an invariant of the serving path regardless of what any screening bound said. The
  alternatives were worse:
  - *keep the supremum* -- honest but inert, and the docs would have to admit the gate cannot
    distinguish candidates, which makes it a budget feasibility check on the arm set rather than a
    promotion gate;
  - *use expected cost* -- discriminates, but understates a candidate whose spend is high-variance
    (concentrated exploration on an expensive arm), which is exactly the candidate the gate is for;
  - *bound the tail by Hoeffding on the per-request range* -- a valid supremum-style bound with no
    distributional assumption, but at these horizons it is dominated by the range, so it degenerates
    back toward the dearest-arm constant.
- **`ScreeningConfig` gained `cost_bound_z`**, and the config is part of the hashed screening report,
  so the conservatism a decision was made under is recorded rather than implied.
- The field names changed (`worst_case_*` to `upper_bound_*`) so the number cannot be read as a
  guarantee it does not provide. The audit payload's key changed with them.
