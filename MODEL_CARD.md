# Model card: bandit-llm-router routing policy

Format after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Every number here,
timings included, was copied from the committed `bench/results/*.json` written by the bench run of
2026-09-28 -- the same run README.md quotes -- on Windows 10 (10.0.19045), Intel64 Family 6 Model 158
(6 logical CPUs), Python 3.12.14, CPU only.

**The one thing to read first:** the quality signal these results are measured against is a
**simulator proxy**, not human or rubric judgement of real model output. Everything below is evidence
that the learning, off-policy evaluation and promotion machinery works and is correctly calibrated
against a world whose ground truth we can compute. None of it is evidence about the real
quality/price frontier of Claude Haiku, Sonnet and Opus. A deployment replaces the quality signal
with a real one, and the *first* question then is whether that signal means what the reward assumes.

## Model details

- **Name:** `routing-policy` (registry model name; the active version is addressed by the `active`
  alias in the content-addressed policy registry).
- **Version:** a policy version string derived from the policy kind and every setting that changes
  its behaviour -- `alpha`/`epsilon`/`posterior_sd`/`mc_samples`, `ridge`, `discount`,
  `propensity_floor`, the featurizer version and the reward configuration. Example from the demo:
  `thompson-94dcfb0210b3`. Two policies that would decide differently cannot share a version.
- **Type:** contextual bandit over 3 discrete arms. Per-arm discounted ridge (Bayesian linear)
  posteriors; the action rule is LinUCB, linear Thompson sampling or linear epsilon-greedy, always
  followed by a propensity floor. Output is a **distribution over arms**, plus the sampled arm and its
  realized propensity.
- **Owner:** Rouviere German. **Date:** 2026-09-28. **License:** MIT (code).
- **Artifact format:** JSON snapshot of the posterior arrays plus hyperparameters, content-addressed
  by SHA-256. Never pickle; npz is loaded with `allow_pickle=False`. A snapshot whose parameters were
  edited is refused when rebuilt.
- **Reproducibility manifest:** every decision log row carries the featurizer version, the policy
  version, the policy *step* (the posterior's update count when the row was written) and the cost
  basis. When the traffic comes from a simulator -- the demo and `bench/bench_router.py` -- the row
  also carries that workload's digest and its seed, passed in at construction; under real callers both
  stay NULL rather than being invented. Every simulation and MLflow run carries the simulator seed, the
  workload digest and the environment digest. `uv run python tasks.py demo` prints both digests, and
  two runs print identical numbers.

## Intended use

- **Primary use:** choosing which of three pinned Claude models serves a given request, so that
  spend falls without a meaningful quality loss, and so that a proposed change to that choice can be
  evaluated on logged data before it is exposed to traffic.
- **Intended users:** the team operating the routing service. Four separate key scopes assume four
  separate roles: callers who route, sources that report rewards, an operator who proposes
  activations, and a different principal who approves them.
- **Out-of-scope uses:**
  - Anything that treats the router's arm choice as a *quality assessment* of a request. It is a
    cost-penalized decision, not a difficulty score, and it is only as good as the reward.
  - Deciding which model to use for a *new* task family the log contains no examples of. The
    posteriors extrapolate linearly in hashed n-gram space, and the screening gate will refuse a
    candidate for lack of evidence, but nothing prevents the *live* policy from routing an unfamiliar
    request confidently and wrongly. Drift monitoring (PSI/KS over the feature columns) is the
    control, and it is a detector, not a guard.
  - Routing where the reward is unobservable, adversarially controlled by the requester, or arrives
    later than the feedback TTL.
  - Any decision with legal, medical, financial or safety consequences attached to which model
    answered.

## The reward, which is the whole objective

```text
reward = clip(quality - cost_weight * min(1, cost / cost_reference), 0, 1)
```

with `cost_weight = 0.5` and `cost_reference_micros = 14_000` by default (ADR 0002). `quality` is in
[0, 1]. The configuration is frozen and contributes to the policy version, so a policy trained under
one trade-off is never screened against logs shaped under another.

`cost_weight` is an exchange rate between quality and dollars. **Changing it changes what "better"
means**, which is why it is versioned rather than tuned. At 0.5 with this cost reference, the router
learns to send ~82% of traffic to Haiku; at 0 it would converge on Opus everywhere. Anyone reading
"36.5% cheaper at equal quality" should read it as "at this exchange rate", and the mean-quality
column is reported next to the saving precisely so a quality regression cannot hide inside a single
reward number.

## Training data

- **Source:** a seeded synthetic workload, `src/bandit_llm_router/simulator/`. Prompts are drawn from
  five families -- extraction, summarization, multi-step reasoning, code, small talk -- each with a
  latent per-request difficulty that shapes the prompt but is never exposed to the policy.
- **Fingerprint:** identified by content, not by timestamp. The bench runs quote their workload and
  environment digests (for example workload `0224abab08862332`, environment
  `37e0003411c3397f`); the digest changes with anything that changes the traffic.
- **Quality model:** per arm, expected quality falls with difficulty, larger arms degrade more slowly,
  and the arms are near-tied on easy work. Observed quality is noisy but bounded in [0, 1]. Costs come
  from `config/pricing.toml` (USD per million tokens, checked against the price table that charges the
  budget ledger).
- **Non-stationarity:** scheduled `QualityShift`s, e.g. a model update that degrades one arm across
  every family at a given step.
- **Contract:** the decision log is validated by a pandera schema (`ml/contracts.py`). Rows with a
  zero or above-one propensity, a reward outside the range, a duplicate decision id, a ceiling below
  the estimate or a malformed identifier are **rejected**, not quarantined. Any extra column fails --
  which is the mechanism that keeps prompts out of the log.
- **Known gaps:** the simulator's expected reward is a smooth function of difficulty and family, so it
  is friendlier than reality in two specific ways: no adversarial or out-of-distribution prompts, and
  no correlation between a request's cost and its difficulty beyond what the arm's token estimates
  imply. It also contains no per-tenant heterogeneity.

## Evaluation

Held out by construction: policies are evaluated on request streams drawn from the pool with separate
seeds from the ones they trained on, and OPE is scored against the simulator's **exact** policy value
(computed by integrating over the quality noise), not against a sampled average.

### Learning and cost

`uv run python bench/bench_regret.py --check` -- 200,000 requests, 111.2s.

| Metric | LinUCB | Thompson | Target |
|---|---:|---:|---|
| Cost saving vs always-Sonnet | 36.5% | 38.9% | >= 30% |
| Mean quality delta vs Sonnet | +0.0048 | -0.0003 | within 1pp |
| Cost saving vs always-Opus | 77.9% | 78.8% | reported |
| In-class regret log-log slope | 0.766 | 0.598 | < 0.8 |
| USD / 1k requests | 2.952 | 2.840 | -- |

Static baselines' slopes sit at 0.996-1.004 (linear regret, as expected). On a well-specified linear
problem the separation is sharper: LinUCB regret 10.3 (slope 0.094), Thompson 43.8 (0.163),
epsilon-greedy 311.6 (0.876), static 7258.3 (0.996).

### Off-policy evaluation

`uv run python bench/bench_ope.py --check` -- 40,000 logged rows, 60 trials
(15 seeds x 4 candidates), 204.8s.

| logging floor | estimator | bias | RMSE | 95% coverage |
|---|---|---:|---:|---:|
| 0.05 | ips / snips / dr | -0.0004 / +0.0001 / -0.0001 | 0.0036 / 0.0022 / 0.0007 | 0.983 / 0.950 / 1.000 |
| 0.01 | ips / snips / dr | -0.0009 / -0.0002 / -0.0003 | 0.0059 / 0.0035 / 0.0011 | 0.967 / 0.933 / 0.983 |

**Calibration caveat 1 -- the intervals are conservative.** DR is the most accurate estimator by RMSE
(5x better than IPS) but its bootstrap intervals are *conservative*: 0.983-1.000 coverage against a
nominal 0.95, above the [0.90, 0.98] band targeted for this project. For a promotion gate the
direction is the safe one -- an over-wide interval refuses candidates rather than admitting bad ones --
but it means the gate is stricter than "95% confidence" suggests, and DR interval widths should not be
quoted as calibrated uncertainty. SNIPS (0.950 / 0.933) is the calibrated estimator.

**Calibration caveat 2 -- these numbers are for a frozen logging policy, and the gate's own log is not
frozen.** The table above comes from a deliberately frozen logger, so its rows are i.i.d. given the
context stream and a bootstrap over rows measures what it claims. `RouterService.screen` does not have
that luxury: it evaluates candidates on the log the live policy wrote while it was learning, where
each row's propensity depends on earlier rewards. The measured cost of that, on the same candidates
and seeds logged twice at 8,000 rows, is **zero** -- DR coverage 0.967 for the frozen logger and 0.967
for the learner, under both the row bootstrap and a block bootstrap over contiguous rows (README
Results has the full table, with the arm-share drift that proves the "online" log really was
adaptive). The reason the estimator survives is that importance weighting is conditionally unbiased,
so each row's term is a martingale difference. Because one simulator is not a proof, the service
resamples contiguous blocks of 250 rows by default and the screen endpoint accepts a `policy_step`
window; genuinely anytime-valid inference for adaptive data is a README non-goal.

Screening decisions, swept over candidate quality: 9/9 truly worse candidates rejected against a
converged incumbent, 9/9 truly better candidates admitted against a stale cost-blind incumbent. The
demo additionally checks its own verdicts against ground truth, in the same two-incumbent shape and
with strict inequalities: every admitted candidate is genuinely better than its incumbent by exact
policy value, and every refused one genuinely worse.

**The cost gate, precisely.** Gate 3 bounds spend over a horizon by charging every logged context at
each arm's *ceiling* price weighted by the candidate's own action distribution, plus a one-sided
deviation term (`cost_bound_z`, default 3 standard deviations of the horizon total). It is therefore a
high-probability bound on total spend, not a supremum, and it *discriminates between candidates*: in
the demo, at a horizon of 100,000 requests, a trained cost-aware policy is priced at 225 USD,
always-Sonnet at 274 and always-Opus at 654, against a 300 USD cap. The supremum -- the dearest
reachable arm on every request -- is reported alongside as `dearest_arm_usd_per_request`, but it cannot
be the gate: every policy here applies a propensity floor, so every arm stays reachable and the
supremum is the same constant for every candidate. Hard bounding of spend is the runtime budget
guard's job, not this bound's: it reserves each call's ceiling against the tenant cap before the
provider is called.

### Adaptation

`uv run python bench/bench_router.py --check` -- 120,000 requests, shift at one third.

| posterior | trough reward ratio | requests to recover 95% of oracle |
|---|---:|---|
| stationary (discount 1.0) | 0.5755 | 19,851 |
| discounted 0.999 | 0.9766 | never fell below target |
| discounted 0.9995 | 0.9684 | never fell below target |

## Robustness, drift and monitoring

- **Adversarial feedback:** with 10% of decisions attacked, mean decision value drops 0.2327 (36%)
  undefended, 0.0002 with clipping only, and 0.0001 with the full aggregation. The honest reading is
  that **bounding the reward range is where nearly all of the protection is**; robust aggregation
  covers attacks that stay inside the valid range.
- **What the *online* learner actually gets, which is weaker:** the full aggregation protects the
  **stored** reward that OPE and offline retraining read. The live posterior learns from the **first
  accepted report only**, so one report means one source and the per-source median and weighted median
  never run -- clipping and the reduced single-source weight are its whole defense. Measured on the
  same poisoned stream: value drop **0.0043**, 43x the full pipeline's 0.0001. Published as its own
  row in `bench/bench_router.py` rather than folded into the number above.
- **Two more limits, both asserted by tests:** a hostile *majority* of sources defeats the median
  (`tests/unit/test_feedback.py::test_a_majority_attacker_wins_and_the_docs_say_so`); and at exactly
  **two** sources the *lower* report wins, because two weights renormalise to `[0.5, 0.5]` and the
  lower weighted median lands on the first sorted value. So a **suppression** attacker reporting
  `reward_min` beats one honest source while an inflation attacker does not, and the
  minimum-independent-sources requirement is **three**
  (`test_with_two_sources_the_lower_report_wins_so_suppression_beats_inflation`).
- **Small logging propensities:** OPE accuracy is measured at floors 0.05 and 0.01 so the effect of
  large importance weights on each estimator is visible rather than assumed.
- **Property-based invariants (hypothesis):** propensities sum to 1 and respect their floor for
  arbitrary contexts and posterior states; the reward stays in [0, 1] for every cost weight and input;
  committed spend never exceeds a tenant cap over random reserve/commit/release sequences.
- **Drift monitors:** PSI and a two-sample KS test over the router's real feature columns
  (`monitoring.py`, `ml/drift.py`), exported as `context_feature_psi`. Default PSI threshold 0.2. The
  KS p-value uses the asymptotic distribution, so it is not exact at small sample sizes.
- **Serving metrics:** `route_decisions_total{arm}`, `spend_usd_total{tenant}`, `estimated_quality`,
  `propensity_min`, `ope_dr_lower_bound`, `ope_ess`, `provider_fallbacks_total`, `breaker_state`,
  `context_feature_psi`, plus an OpenTelemetry span per request. `propensity_min` is the one to alert
  on: if it drifts toward zero, the log is quietly becoming unusable for OPE and every later screening
  decision rests on less evidence than it appears to.
- **Retraining / re-screening trigger:** a PSI breach on the feature distribution, a sustained drop in
  the trailing reward ratio, or a provider model update. Retraining produces a *candidate*; it reaches
  traffic only through the gate.

## Ethical considerations

- **Privacy.** Prompts are the sensitive asset and are never persisted: they reach the featurizer and
  stop there, and the only prompt-derived value stored is a keyed BLAKE2b digest whose key makes
  digests non-enumerable. Hashed character n-gram features are not prompt text but are derived from
  it, so the feature matrix should be treated as sensitive. Short, low-entropy prompts could be
  confirmed by guessing if the digest key leaked.
- **Whose quality is being optimized.** A single scalar reward averaged over traffic can improve the
  mean while getting systematically worse for a minority of requests -- a rare prompt family, a
  non-English request, one tenant's workload. Nothing in this repo slices reward by population, and
  the simulator has no such structure to find. A deployment should report reward and quality by slice
  before trusting an aggregate improvement.
- **Whose money.** The router optimizes cost for the operator. If the party bearing the cost is not
  the party bearing a lower-quality answer, the trade-off encoded in `cost_weight` is being made on
  someone else's behalf, and should be disclosed rather than buried in a config file.
- **Feedback as a power relation.** Whoever supplies rewards defines quality. The defenses here bound
  what a *hostile minority* can do; they cannot make a bad reward definition good.

## Caveats and recommendations

1. **Simulated quality is the binding limitation.** Do not quote the cost-saving numbers as evidence
   about real models. Re-measure with a real quality signal before believing any of them.
2. **Validate the reward before the policy.** The gate can prove a candidate is better *at the reward
   as defined*. If the reward is a poor proxy for what users want, the gate will faithfully promote
   policies that make things worse.
3. **OPE assumes no unobserved confounding.** If anything upstream already routes on information the
   log does not contain, every estimate here is biased in a way the bench cannot detect.
4. **The logged propensity is load-bearing.** Any change to the serving path that alters action
   probabilities without altering the logged propensity invalidates the log silently. This is why the
   breaker and budget restrictions log the *realized* distribution (ADR 0005), and why
   `propensity_min` is a monitored gauge.
5. **DR intervals are conservative, not calibrated.** Use SNIPS if you need a calibrated interval;
   keep DR for the point estimate and the gate.
6. **One activation is atomic.** There is no traffic split and no automatic rollback. The circuit
   breaker and per-tenant caps bound the damage of a bad activation; nothing undoes it automatically.
