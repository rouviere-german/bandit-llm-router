# bandit-llm-router

A routing service that learns which Claude model should answer each individual request, and refuses
to put a new routing policy into production until the recorded evidence says it beats the one already
running.

## The problem

Every product built on a large language model answers the same question on every request: which
model? The big one writes the best answers and costs the most; the small one is cheap and sometimes
not good enough. In the price list this repo routes against, an Opus output token costs five times a
Haiku output token. Most teams pick one model in a config file and leave it there.

That choice fails in three ways, and each one costs something real.

- **You overpay for easy work.** "Summarize these three bullets" goes to the most expensive model at
  the same price as a hard refactor. In the simulation below, sending everything to Opus costs
  **$13.38 per thousand requests**; the learned router costs **$2.95** for the same traffic.
- **You underpay for hard work.** Pin everything to the cheap model and the requests that needed the
  strong one come back weak. The user sees a bad answer; nobody sees a log line.
- **You cannot safely change your mind.** A new routing rule can only be judged by running it, so the
  experiment *is* the exposure. If the rule is worse, real users get worse answers and the bill moves
  while you find out, and the rollback argument is a guess.

The people this hurts are whoever signs off the model bill, the user whose hard question was sent to
the cheap model, and the engineer asked whether last week's routing change is why quality dropped,
with nothing but a dashboard to answer from.

This repo makes model choice a **learned decision policy** instead of a configuration value, records
each decision in a form that can honestly be re-analysed later, and puts a gate in front of
promotion: a candidate policy has to beat the incumbent on the historical record, and a second person
has to approve it, before it serves a single request.

Everything here runs offline on a laptop CPU. A seeded simulator and a fake provider make routing,
learning, evaluation and the whole approval path reproducible with no API key and no network.

## What this does

- **Picks a model per request.** The router reads the incoming prompt's shape -- length, structure,
  the kind of task it looks like -- and chooses between Haiku, Sonnet and Opus for that one request.
- **Learns from what came back.** Ratings arrive later, through a separate endpoint, and move the
  router's beliefs. Nobody hand-writes routing rules, and nobody retunes them when a model changes.
- **Records how likely each choice was.** Every decision is stored with its *propensity* -- the
  probability that choice had of being made. That one number is what makes the next bullet possible.
- **Scores a new policy on old traffic, before it runs.** This is *off-policy evaluation*: estimating
  what a policy would have earned from decisions it never made. Three estimators, with confidence
  intervals, answer "would this have been better?" without exposing anyone to it.
- **Refuses to promote a policy that has not earned it.** A candidate must clear three checks on the
  evidence and then be approved by a different person than the one who proposed it. Every promotion,
  and every refusal, is written to a tamper-evident log.
- **Keeps a hard lid on spend and survives hostile feedback.** Per-customer dollar caps are enforced
  before the model is called, not after. Ratings are treated as untrusted input, because whoever can
  submit them can steer where the money goes.

## Architecture

The picture first, then the same thing in words.

```mermaid
flowchart TB
    subgraph serve["Serving path: one request"]
        req["POST /v1/route<br/>tenant, prompt, max_output_tokens"] --> feat["Featurizer<br/>hashed char n-grams + length,<br/>structure, metadata"]
        feat -.->|"prompt stops here"| stop(["never persisted"])
        feat --> pol["Live policy<br/>LinUCB / Thompson / eps-greedy<br/>+ propensity floor"]
        pol -->|"a probability for every arm"| restrict["Restrict<br/>breaker open? ceiling affordable?<br/>mass moves along fallback order"]
        restrict --> reserve[("Tenant budget guard<br/>reserve the worst-case cost<br/>BEFORE the call")]
        reserve --> prov{{"LLMProvider"}}
        prov --> fake["FakeProvider (offline, seeded)"]
        prov --> claude["AnthropicProvider (lazy SDK,<br/>only with a key)"]
        prov --> commit["Commit real cost from usage"]
        commit --> log[("Decision log (DuckDB)<br/>features, arm, REALIZED propensity,<br/>policy version, cost")]
    end

    subgraph learn["Learning: ratings arrive later"]
        fb["POST /v1/feedback<br/>separate key scope"] --> ing["Ingestor<br/>clip, rate limit, dedupe,<br/>per-source median then<br/>weighted median"]
        ing --> log
        ing -->|"first accepted report only"| pol
    end

    subgraph gov["Governance: a policy cannot deploy itself"]
        log --> ope["IPS / SNIPS / DR<br/>cross-fitted ridge reward model<br/>bootstrap CIs, ESS"]
        ope --> screen{"Eligible?<br/>DR lower bound beats incumbent<br/>ESS at or above minimum<br/>horizon cost bound within cap"}
        screen -->|no| refuse["Refused, and audited"]
        screen -->|yes| propose["propose"]
        propose --> approve["approve<br/>(a different principal)"]
        approve --> activate["activate<br/>re-checks the evidence"]
        activate --> pol
        activate --> audit[("Hash-chained<br/>activation audit")]
        activate --> reg[("Content-addressed<br/>policy registry + MLflow")]
    end
```

### How a request flows, step by step

1. **A request arrives** at `POST /v1/route` with a customer id, a prompt and a ceiling on output
   tokens.
2. **The prompt is turned into numbers and then discarded.** The featurizer produces hashed character
   n-grams plus length, structure and metadata. The prompt text goes no further; the log has no
   column that could hold it.
3. **The policy returns a probability for each model**, never a bare pick. No model ever falls below
   the *propensity floor* -- a minimum probability that keeps every option being measured.
4. **Unavailable models are removed.** If a model's circuit breaker is open (it has been failing) or
   its worst-case cost no longer fits the customer's remaining budget, its probability is set to zero
   and its share moves to the first healthy, affordable alternative.
5. **One model is sampled from what remains, and its worst-case cost is reserved** against the
   customer's cap *before* any call is made. If the ceiling does not fit, the call does not happen.
6. **The provider answers** -- the offline fake provider by default, Anthropic when
   `ANTHROPIC_API_KEY` is set. The real cost is committed from the reported token usage and the
   unused part of the reservation is released.
7. **One row is written to the decision log:** features, the model chosen, the probability it
   *actually* had after step 4, the policy version and the cost.
8. **Ratings arrive later** on a different endpoint with a different key, are clipped, rate limited
   and deduplicated, and are combined across sources. The first accepted report for a decision also
   updates the live policy.
9. **When someone proposes a new policy**, the gate re-reads that log, estimates the candidate's
   value three ways with confidence intervals, and returns a yes or no with reasons attached.

Layout:

```text
src/bandit_llm_router/
  simulator/     workload families, per-arm quality model, scheduled shifts
  policies/      discounted ridge posteriors; LinUCB, Thompson, eps-greedy, static, keyword rule
  ope/           IPS / SNIPS / DR estimators, bootstrap CIs, ESS; the screening gate
  features.py  reward.py  arms.py  budgets.py  breaker.py  feedback.py  logstore.py
  activation.py  service.py  experiments.py  monitoring.py  demo.py
  core/          settings, redacting logs, tracing, metrics, hardened app   ml/  agentic/
bench/  4 benches + thresholds.json      tests/unit, tests/integration
docs/THREAT_MODEL.md  docs/adr/  CLAIMS.md  MODEL_CARD.md
```

## Results

Every number below was copied from the committed `bench/results/*.json`, written by one bench run on
2026-09-28 with `--check --update-reference`. Machine: Windows 10 (10.0.19045), Intel64 Family 6
Model 158 (6 logical CPUs), Python 3.12.14, CPU only, no API key. Each bench enforces its own section
of `bench/thresholds.json` and every one printed "all ... within thresholds". CI runs the same
commands (jobs `bench` and `evaluation`). MODEL_CARD.md quotes the same run, including its timings.

The simulations are seeded, so the reward, cost, regret, OPE and robustness figures reproduce exactly
on a rerun; only the timings move.

| What was measured | Result | Command |
|---|---|---|
| Cost against always-Sonnet at matched quality | **36.5% cheaper** (LinUCB), 38.9% (Thompson) | `uv run python bench/bench_regret.py --check` |
| Accuracy of the estimator the promotion gate trusts | DR error (RMSE) **0.0007** against IPS's 0.0036 | `uv run python bench/bench_ope.py --check` |
| Gate decisions on candidates of known quality | **9 of 9** genuinely worse refused, **9 of 9** genuinely better admitted | `uv run python bench/bench_ope.py --check` |
| Recovery after a model silently degrades | discounted posteriors **never dropped below target**; the stationary one took **19,851 requests** | `uv run python bench/bench_router.py --check` |
| Damage from 10% hostile ratings | **0.2327** of mean value lost undefended, **0.0001** with the full pipeline | `uv run python bench/bench_router.py --check` |
| Router overhead per request, no model call | p50 **24.8 ms** with the shared budget file, **6.0 ms** in memory | `uv run python bench/bench_router.py --check` |

### Learning and cost, 200,000 simulated requests

`uv run python bench/bench_regret.py --check` (111.2s). All policies see the *same* request stream.
*Regret* is how much reward a policy gave up against the best achievable choice; `in-class slope` is
the fitted log-log slope of that shortfall over time, and a slope below 1 means the gap is still
closing rather than growing in proportion to traffic.

| policy | mean reward | mean quality | USD / 1k req | saving vs Sonnet | saving vs Opus | quality delta vs Sonnet | in-class regret | in-class slope | arm shares (h/s/o) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **linucb** | 0.6508 | 0.7556 | 2.952 | **36.5%** | 77.9% | +0.0048 | 468.0 | **0.766** | 0.82 / 0.10 / 0.07 |
| **thompson** | 0.6497 | 0.7506 | 2.840 | **38.9%** | 78.8% | -0.0003 | 702.2 | **0.598** | 0.83 / 0.10 / 0.07 |
| epsilon_greedy | 0.6454 | 0.7503 | 2.955 | 36.4% | 77.9% | -0.0006 | 1541.8 | 0.938 | 0.82 / 0.10 / 0.07 |
| keyword_rule | 0.6235 | 0.7782 | 4.338 | 6.7% | 67.6% | +0.0273 | 6797.3 | 0.996 | 0.59 / 0.26 / 0.15 |
| always haiku | 0.6416 | 0.7144 | 2.053 | 55.8% | 84.6% | -0.0365 | 2299.5 | 1.004 | -- |
| always sonnet | 0.5851 | 0.7509 | 4.648 | 0 | 65.2% | 0 | 14498.1 | 0.998 | -- |
| always opus | 0.4048 | 0.8730 | 13.375 | -187.7% | 0 | +0.1221 | 50032.2 | 0.999 | -- |

**What that means.** The learned router served the same traffic as always-Sonnet for 36.5% less money
at the same measured quality -- +0.48 of a percentage point, a rounding difference. Always-Haiku is
cheaper still, and *that is the honest comparison this table exists to make*: it buys its saving with
a 3.7-point quality drop, which is why it scores worse once cost and quality are weighed together.
The hand-written keyword rule -- the thing a team would actually write first -- saves 6.7%. The fixed
choices have slopes of about 1.0, meaning their shortfall grows in proportion to traffic forever, as
it must: they never learn anything.

On a **well-specified** linear problem (60,000 requests), where the model class contains the truth,
the learners separate cleanly from the alternatives:

| policy | regret | regret slope |
|---|---:|---:|
| linucb | 10.3 | 0.094 |
| thompson | 43.8 | 0.163 |
| epsilon_greedy | 311.6 | 0.876 |
| static arm | 7258.3 | 0.996 |

### Off-policy evaluation accuracy, 40,000 logged rows

`uv run python bench/bench_ope.py --check` (204.8s). Estimates are scored against the simulator's
*exact* policy values, over 60 trials (15 seeds x 4 candidates), at two logging propensity floors.

| logging floor | estimator | bias | RMSE | 95% CI coverage | mean ESS |
|---|---|---:|---:|---:|---:|
| 0.05 | ips | -0.0004 | 0.0036 | 0.983 | 15345.6 |
| 0.05 | snips | +0.0001 | 0.0022 | 0.950 | 15345.6 |
| 0.05 | **dr** | -0.0001 | **0.0007** | 1.000 | 15345.6 |
| 0.01 | ips | -0.0009 | 0.0059 | 0.967 | 9857.2 |
| 0.01 | snips | -0.0002 | 0.0035 | 0.933 | 9857.2 |
| 0.01 | **dr** | -0.0003 | **0.0011** | 0.983 | 9857.2 |

**What that means.** The question is how close an estimate lands to a truth we can only see because
this is a simulator. The doubly robust estimator (DR) misses by 0.0007 on a reward scale of 0 to 1 --
about five times closer than the plain importance-weighted estimator -- and the ordering holds when
the logging floor drops to 0.01, where the arithmetic gets harder. "95% CI coverage" is how often the
interval contained the truth, against a target of 95%.

**Coverage is conservative, not calibrated.** DR's bootstrap intervals cover the truth 98.3-100% of
the time against a nominal 95%. That is the safe direction for a promotion gate -- an over-wide
interval refuses candidates it should have admitted rather than admitting ones it should not -- but
it is not the `[0.90, 0.98]` band this project set out to hit, and it means the gate is *stricter*
than 95% implies. SNIPS (0.950, 0.933) is the well-calibrated one. See Non-goals.

What the gate did with those estimates, swept over candidates of known quality:

| sweep | truly worse candidates | rejected | truly better candidates | admitted |
|---|---:|---:|---:|---:|
| vs a converged incumbent | 9 | **9 (100%)** | 0 | -- |
| vs a stale cost-blind incumbent | 0 | -- | 9 | **9 (100%)** |

Both sweeps exist because one alone proves nothing: against a good incumbent nothing on offer is
genuinely better, so a lone sweep could only ever measure the gate's willingness to say no.

#### The coverage above is for a *frozen* logging policy. The shipped gate's log is not frozen.

`RouterService.screen` evaluates candidates on the decision log the live policy wrote **while it was
learning**: every row's propensity depends on the rewards of the rows before it, so the rows are not
independent, and a bootstrap that resamples single rows is assuming they are. Filtering cannot fix it
-- `policy_version` hashes hyperparameters, so one learning LinUCB keeps one version for its whole
life.

So it is measured rather than argued. The same 4 candidates and 15 seeds, logged twice at 8,000 rows,
once frozen and once by a learner starting cold, with the DR interval computed both ways on
*identical* data:

| logging policy | bootstrap | DR 95% coverage | mean CI width | RMSE | arm-share drift |
|---|---|---:|---:|---:|---:|
| frozen | rows | 0.967 | 0.0115 | 0.0021 | 0.023 |
| frozen | blocks of 250 | 0.967 | 0.0114 | 0.0021 | 0.023 |
| online learner | rows | 0.967 | 0.0116 | 0.0020 | 0.069 |
| online learner | blocks of 250 | 0.967 | 0.0113 | 0.0020 | 0.069 |

**The measured degradation on this workload is zero**, and that is the honest headline: 0.967 in all
four cells. The theoretical concern is real -- and the reason importance weighting survives it is
that it is conditionally unbiased, so each row's term is a martingale difference and the naive
variance estimate stays roughly right. `arm-share drift` is the non-vacuity check: the total-variation
distance between the first and last tenth of each log's arm shares, 0.069 for the learner against a
0.023 sampling-noise floor for the frozen logger, and `bench/thresholds.json` fails the bench if the
"online" log stops being adaptive. Two mitigations ship anyway, because a measurement on one simulator
is not a proof: the service's bootstrap resamples contiguous blocks of 250 rows by default
(`APP_SCREENING_BOOTSTRAP_BLOCK_SIZE`), and `POST /v1/policies/{digest}/screen` takes
`min_policy_step` / `max_policy_step` so an operator can screen on a stretch of the log over which the
posterior barely moved.

### Adaptation after a mid-run model update, 120,000 requests

`uv run python bench/bench_router.py --check` (151.2s). A third of the way through the run, Haiku --
which is carrying about 82% of the traffic -- gets worse at every kind of prompt, with no
announcement. "Recover" means getting the trailing reward back to 95% of what perfect routing would
have earned.

| posterior | discount | pre-shift ratio | trough ratio | requests to recover | final ratio |
|---|---:|---:|---:|---|---:|
| stationary | 1.0 | 0.9969 | **0.5755** | 19,851 | 0.9973 |
| discounted | 0.999 | 0.9895 | 0.9766 | **never fell below target** | 0.9980 |
| discounted | 0.9995 | 0.9928 | 0.9684 | **never fell below target** | 0.9982 |

**What that means.** The textbook version of this router treats a year-old observation as worth
exactly as much as this morning's. When the model it trusted quietly got worse, it fell from 99.7% of
perfect routing to 57.6% and took nearly 20,000 requests to climb back -- because its confidence had
collapsed precisely where the evidence had gone stale. Ageing old evidence out fixes it: both
discounted posteriors absorbed the same shift without ever dipping below target. The honest reading
of this table is "19,851 versus never dipped", not two recovery counts (ADR 0004). The bench asserts
that the stationary policy really is hurt, so this comparison cannot quietly become vacuous.

### Robustness to feedback poisoning, 40,000 requests

10% of decisions receive hostile reward reports, replayed identically through four channels.

| feedback | defenses | mean decision value | value drop |
|---|---|---:|---:|
| honest only | -- | 0.6495 | 0 |
| 10% adversarial | none | 0.4168 | **0.2327** |
| 10% adversarial | clipping only | 0.6493 | 0.0002 |
| 10% adversarial | full aggregation (the **stored** reward) | 0.6494 | 0.0001 |
| 10% adversarial | online posterior (first report, clipped) | 0.6452 | **0.0043** |

**What that means.** If one rater in ten lies about how good an answer was, an undefended router
loses 0.2327 of its 0.6495 mean value -- over a third, roughly the gap between a learned router and
always-Opus. Simply refusing to believe ratings outside the valid range recovers nearly all of it,
and that is the honest finding: **bounding the reward range is where almost all of the benefit is.**
The robust aggregation earns the remaining 0.0001 plus protection against attacks that clipping
cannot see, namely many reports that are each individually plausible.

The last row exists because the third does not describe the deployed learner. The full aggregation
protects the **stored** reward -- what evaluation and offline retraining read -- while
`submit_feedback` teaches the **online** posterior from the *first accepted report only* (an online
ridge update cannot be withdrawn, so applying several would double-count one outcome). One report is
one source, so the per-source median and the weighted median never run on that path: clipping and the
reduced single-source weight are its whole defense, and its measured drop is 43 times the full
pipeline's. Two more limits, both asserted by tests rather than implied: a hostile **majority** of
sources wins, and at exactly **two** sources the *lower* report wins, so a suppression attacker beats
one honest source while an inflation attacker does not -- the minimum-independent-sources requirement
is three.

### Routing overhead, 2,000 requests through the live path

Featurize, decide, restrict, reserve, call the offline provider, commit, log to DuckDB. This is the
router's own overhead with no real model call in it; a live call to a hosted model is orders of
magnitude slower. Reported twice because the shared budget file dominates one of the numbers
(ADR 0006); both configurations logged all 2,000 decisions.

| budget guard | p50 (us) | p95 (us) | p99 (us) |
|---|---:|---:|---:|
| shared budget file (default) | 24,817 | 31,514 | 48,669 |
| in-memory guard | 6,000 | 12,233 | 30,499 |

**What that means.** With the default multi-worker budget file, half of requests spent under 24.8 ms
inside the router and the slowest 1 in 100 still finished in 48.7 ms. Holding the budget in memory
instead cuts the median to 6.0 ms. The ~19 ms difference is the price of correctness: two atomic
writes plus a reload under a file lock, per reservation and per commit, so several worker processes
can draw down one shared cap without overspending it. That is a deliberate trade of latency for a
budget that stays correct across processes, and on this machine it is disk-bound.

### Core micro-benchmarks

`uv run python bench/bench_baseline.py --check`.

| benchmark | runs | p50 (us) | p95 (us) |
|---|---:|---:|---:|
| redact_text_1kb | 2000 | 339.5 | 425.4 |
| rate_limiter_check | 20000 | 1.9 | 2.0 |
| llm_request_sha256_10_messages | 2000 | 98.3 | 126.9 |
| audit_append_fsync | 200 | 7857.1 | 12350.0 |
| audit_verify_1000_entries | 10 | 34104.4 | 36157.2 |
| drift_check_10k_vs_10k | 30 | 2609.4 | 3194.6 |

`audit_append_fsync` includes one `fsync` per entry, so it mostly measures the disk. These are
single-machine numbers for spotting regressions, not performance claims.

### Test suite

`uv run python tasks.py test` reports **97.90% line-and-branch coverage** against an 85% floor
(`coverage.xml`: 5307 of 5389 lines and 989 of 1042 branches). Property-based tests (hypothesis)
cover the invariants themselves rather than examples of them: propensities sum to 1 and respect their
floor; the reward stays in [0, 1] for every cost weight; committed spend never exceeds a cap over
random reserve/commit/release sequences.

## How it works

| Component | Responsibility |
|---|---|
| `simulator/` | Seeded prompt families with latent difficulty, a per-model quality response, and scheduled quality shifts. This is the ground truth every measurement above is scored against. |
| `features.py` | Hashing vectorizer over character n-grams plus length, structure and metadata. No embedding downloads, no network, and the prompt is not retained. |
| `policies/` | Discounted ridge posteriors shared by LinUCB, Thompson sampling and epsilon-greedy, plus the static and keyword baselines they are measured against. |
| `budgets.py`, `breaker.py` | Per-tenant USD caps enforced before the call, and a circuit breaker whose fallback order comes from `config/pricing.toml`. |
| `logstore.py` | DuckDB decision log: features, action, realized propensity, policy version, cost. Schema-level refusal of any prompt-shaped column. |
| `feedback.py` | Untrusted ratings: clip, rate limit, dedupe, per-source median then weighted median. |
| `ope/` | IPS, SNIPS and doubly robust estimators with a cross-fitted ridge reward model, bootstrap confidence intervals, effective sample size, and the three-gate screen. |
| `activation.py` | Propose / approve / activate, with a hash-chained audit log and a content-addressed policy registry. |
| `service.py`, `core/` | FastAPI app, four API-key scopes, redacting logs, OpenTelemetry spans, Prometheus metrics, `/healthz` and `/readyz`. |

Three decisions carry most of the weight.

**The propensity that is logged is the one that ran.** Arms blocked by an open breaker or by a budget
ceiling are zeroed and their probability mass moves to the first affordable, healthy arm; a
post-failure redirect logs its adjusted propensity and flags the row. The obvious alternative --
logging the distribution the policy *intended* -- is simpler and quietly poisons every later estimate,
permanently, because the estimator divides by a number that was never true (ADR 0005).

**Exploration is priced before it happens, not budgeted after.** The guard admits a call only if the
arm's cost *ceiling* still fits the tenant's remaining cap, so `spent <= cap` is an invariant rather
than an alert, and an arm the policy wants is simply unavailable instead of overdrawn. Screening
applies the same idea over a horizon: every logged context is charged at the ceiling price of each
arm, weighted by the candidate's own action distribution, plus a one-sided deviation term, and a
candidate whose bound busts the budget is refused however good it looks. This is a high-probability
bound on total spend, not a supremum, and it is per candidate by construction -- the supremum ("the
dearest reachable arm every request") is also reported, but it cannot be the gate, because a
propensity floor leaves every arm reachable and collapses it to one constant for every candidate
(ADR 0007).

**Thompson sampling is deployed as its Monte-Carlo action distribution.** A sampled argmax has no
closed-form propensity; logging it as if the action had been certain would silently invalidate the
log, which is worse than not logging at all (ADR 0003).

Two further trade-offs worth naming:

- **A cost-penalized scalar reward instead of a constrained objective** (ADR 0002). The textbook
  statement is "maximize quality subject to spend below B", solved with a learned Lagrange
  multiplier. This repo folds cost into a bounded, versioned reward and keeps hard spend limits as a
  *separate* runtime mechanism. A learned multiplier means the exchange rate between quality and
  dollars drifts on its own, and a policy trained under one exchange rate can then be screened
  against logs shaped by another. Freezing it makes changing it a governance event rather than a
  hyperparameter sweep, and the hard cap is enforced by code that cannot be argued with.
- **One discounted, re-regularized Gram matrix for non-stationarity** (ADR 0004). With `discount = 1`
  it is exactly LinUCB; below 1, evidence decays with a half-life and uncertainty grows again in
  directions that stop being refreshed, so exploration restarts on its own. The
  `(1 - discount) * ridge * I` term is the part that is easy to omit and costly to omit: without it
  the posterior decays toward zero rather than toward the prior, and confidence widths diverge
  instead of merely widening.

Architecture decision records: [0001 recording decisions](docs/adr/0001-record-architecture-decisions.md),
[0002 cost-penalized reward](docs/adr/0002-cost-penalized-scalar-reward.md),
[0003 Thompson propensities](docs/adr/0003-thompson-sampling-logs-monte-carlo-propensities.md),
[0004 discounted ridge](docs/adr/0004-discounted-ridge-for-non-stationarity.md),
[0005 realized propensity](docs/adr/0005-log-the-realized-propensity-after-restriction.md),
[0006 overhead reported twice](docs/adr/0006-report-route-overhead-twice.md),
[0007 pricing exploration](docs/adr/0007-price-exploration-against-reachable-mass.md),
[0008 screening an adaptive log](docs/adr/0008-screen-on-an-adaptive-log-and-say-so.md).

## Security

A router that learns from the outside world has two unusual weak points: the ratings it learns from,
and the money it spends. Both get most of the attention here. The full model, control by control with
the test that proves each one, is in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md); every claim on this
page maps to evidence in [CLAIMS.md](CLAIMS.md); reporting is in [SECURITY.md](SECURITY.md).

**Someone lying about answer quality could steer your spending.** Whoever can submit feedback can
teach the router that the expensive model is the good one. *Mechanism:* reports are clipped into the
valid reward range and counted, rate limited per key, capped per decision and per source,
deduplicated by `report_id`, and combined with a per-source median followed by a weighted median
whose weights are capped *strictly* below one half (a cap of exactly one half is refused, because the
weighted median would then be owned by whoever reaches half the weight first). A single-source reward
learns with reduced weight. The measured worth of all this is the poisoning table above. Three
limits, each asserted by a test rather than implied: **a hostile majority of sources wins**; at
exactly **two** sources the *lower* report wins, so suppression succeeds where inflation fails and the
real requirement is three independent sources; and the aggregation protects the **stored** reward, not
the **online** posterior, which learns from one clipped report.

**Someone could run up your model bill.** *Mechanism:* per-tenant USD caps are enforced *before* the
provider call, against the arm's worst-case cost, so `spent <= cap` is a property rather than a hope;
a failed call refunds its reservation. Screening bounds exploration over a horizon the same way.

**Your users' prompts could leak through the logs.** *Mechanism:* prompts reach the featurizer and
stop. The log's schema has no column that can hold text, the store refuses to open against a
prompt-shaped column, and the only prompt-derived value stored is a keyed BLAKE2b digest whose key
makes digests non-enumerable.

**A bad policy could be pushed with no evidence and no review.** *Mechanism:* screening runs on the
service's own logs (the caller cannot supply them), a report is hash-bound to its candidate *and* its
log batch, the proposer cannot approve their own activation, approvals and activations cannot be
replayed, and every step -- refusals included -- is appended to a hash-chained audit log.

The rest, briefly:

- **Four key scopes** -- route, feedback, admin, approve -- held as SHA-256 digests and compared
  against every digest with no early exit. Separation of duties in the activation gate is only
  meaningful if these are different keys, so they are separate settings.
- **Pinned model allowlist.** Arms come from `config/pricing.toml`, whose models must be on the
  allowlist in `arms.py` and whose prices must agree with the table that charges the budget. Adding a
  model is a code review, not a config push.
- **Provider key from the environment only, never logged.** One test drives routing, feedback and
  governance with sentinel provider, digest and API keys configured, captures the real configured log
  stream, and asserts no sentinel appears in it. gitleaks and semgrep run in CI.
- **Artifacts are pickle-free**; the registry verifies content hashes on load.
- **CI:** ruff, mypy --strict, pytest with an 85% coverage floor, bandit, semgrep with repo rules,
  pip-audit, gitleaks, hadolint, trivy image scan, syft CycloneDX SBOM, the baseline check, and all
  four benches against their thresholds. Base images are pinned by digest.

## Quickstart

Try it in two commands. Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12.
**No API key, no network, no cost.**

```bash
uv sync --frozen
uv run python tasks.py demo
```

The demo runs one simulated day end to end in about 50 seconds and prints seven sections:

1. **Learning** -- the three learners and four baselines over one request stream, with cost, quality
   and regret side by side.
2. **Adaptation** -- a mid-day shift where Haiku degrades, and how much faster the discounted
   posterior notices.
3. **Off-policy evaluation** -- each estimator's error against the simulator's exact values, then
   every candidate's true value next to its estimate and confidence interval.
4. **Screening, approval and activation** -- four candidates go to the gate; the two genuinely better
   ones are admitted and one is really activated, the two worse ones are refused with reasons
   (`dr_lower_bound_does_not_beat_incumbent`, `cost_upper_bound_exceeds_budget`). Each verdict is
   printed next to the candidate's *exact* value, so you can check the gate was right. Two sweeps run
   here, against a stale cost-blind incumbent and against a converged one, because under a
   cost-penalized reward nothing is genuinely worse than always-Opus and a single sweep against that
   incumbent could only ever refuse a copy of itself. `tests/integration/test_cli_demo.py` asserts
   every admission is genuinely better and every refusal genuinely worse, strictly.
5. **Adversarial feedback** -- the poisoning table, regenerated live.
6. **The live path** -- real routing on DuckDB with two tenants, where the small tenant is refused
   mid-run because its $0.004 cap has run out.
7. **An MLflow run** -- parameters and metrics logged to a local file backend.

Because it is seeded, the demo's tables are stable across runs; because it runs at a smaller scale
than the benches, its numbers are its own and differ from the Results section above.

Other entry points:

```bash
uv run python tasks.py check   # ruff, mypy --strict, pytest + coverage, bandit, pip-audit, baseline
uv run python tasks.py bench   # all four benches against bench/thresholds.json (~10 min)
```

A bench run writes its JSON to `bench/results/latest/`, which is gitignored, so the quickstart leaves
a clean working tree. The tracked `bench/results/*.json` is the single run every table above was
copied from; `--update-reference` on an individual bench is what rewrites it.

On Windows, or any machine where the repo drive is small, keep the environment elsewhere:
`$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/bandit-llm-router"`.

`tasks.py` is the single source of truth; the `Makefile` delegates to it so Linux and CI agree with
Windows. Targets: `lint format typecheck test bench demo bandit pip-audit audit baseline check`.

Run the service yourself:

```bash
uv run bandit-llm-router generate-api-key --scope route      # and again per scope
APP_ROUTE_API_KEY_SHA256=<digest> APP_FEEDBACK_API_KEY_SHA256=<digest> \
APP_ADMIN_API_KEY_SHA256=<digest> APP_APPROVER_API_KEY_SHA256=<digest> \
  uv run bandit-llm-router serve

curl -s -H "X-API-Key: <route key>" -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "prompt": "Summarize these release notes in three bullets.",
       "max_output_tokens": 400}' \
  http://127.0.0.1:8000/v1/route
```

With no `ANTHROPIC_API_KEY` the arms are served by the offline provider, which reports token counts
scaled to each arm's own estimate -- so budgets, costs and metrics exercise realistic numbers with no
spend. Set the key to route to the real models.

Verify an activation chain: `uv run bandit-llm-router audit verify var/demo/service/activations.audit.jsonl`.

Containers: `docker compose up --build` runs the non-root, read-only image with a writable volume for
the decision log and audit chain. Image build, smoke test, hadolint, trivy and the SBOM run in CI
(`.github/workflows/ci.yml`, job `docker`); docker was **not** run locally while building this repo.

## Non-goals

What this is *not*. Honest scope limits, not a wish list.

- **Real quality measurement.** Offline `quality` is a simulator proxy with a written-down functional
  form. Every number above validates the learning, evaluation and governance machinery; none of them
  is evidence about a real quality/price frontier. See [MODEL_CARD.md](MODEL_CARD.md).
- **Valid inference on adaptively collected logs.** The coverage this repo measures is for a *frozen*
  logging policy. The shipped gate screens on the log the live learner wrote, which is not i.i.d. in
  its rows; the degradation measures zero on this workload (0.967 either way, table above), the
  service resamples contiguous blocks by default and the screen endpoint takes a `policy_step` window,
  but none of that is a validity *proof* for adaptive data. Anytime-valid confidence sequences, or a
  martingale concentration bound in place of the bootstrap, are the real answer and are not done here.
- **A hard supremum on exploration cost.** Gate 3 is a high-probability bound on horizon spend (the
  arms' ceiling prices under the candidate's own distribution plus three standard deviations of the
  horizon total), not a worst case that cannot be exceeded. The true supremum is reported alongside it
  but cannot be the gate, because a propensity floor keeps every arm reachable and makes it the same
  constant for every candidate. What actually hard-bounds spend is the runtime tenant guard, which
  reserves each call's ceiling before the provider is called.
- **Corroboration before the online update.** The live posterior learns from the first accepted report
  for a decision, so it gets clipping and a reduced single-source weight but never the per-source
  median or the weighted median; the poisoning table publishes that channel's own number rather than
  the stronger one. Holding a decision open for a short window so several sources can vote before the
  first online update would close the gap and is not implemented.
- **Calibrated DR intervals.** DR's bootstrap coverage measures 0.983-1.000 against a nominal 0.95 --
  conservative, outside the `[0.90, 0.98]` band originally targeted. The gate is therefore stricter
  than nominal, which is the safe direction, but tightening the interval (studentized or BCa
  bootstrap) is not done here. SNIPS is the calibrated estimator if you need one.
- **Traffic-splitting rollout.** The gate promotes a policy atomically. Progressive exposure,
  shadow traffic and automated rollback are out of scope; the breaker and the budget caps are what
  bound the damage of a bad activation.
- **A non-stationary regret bound.** The discounted posterior is a simplified form of the weighted
  linear bandit (ADR 0004) without the second matrix its confidence bound needs, so the adaptation
  claim is measured behaviour on a simulator whose shift we wrote, not a proved bound.
- **Reward-model-free evaluation.** DR's reward model is a cross-fitted ridge regression. A
  misspecified reward model shows up as variance, not bias, only because the propensities are right;
  both being wrong is not something OPE can detect.
- **Unobserved confounding.** Every OPE guarantee assumes the logged features contain everything that
  influenced both the action and the reward. Deployed behind another router that already selects on
  hidden information, this breaks silently.
- **Feature-gaming defenses.** A caller who knows the featurizer can pad a trivial prompt so it looks
  hard and get the expensive arm. The cost lands on their own tenant budget, and per-tenant spend and
  arm-share metrics would reveal it, but nothing here detects it.
- **Distributed anything.** Rate-limit buckets, the breaker and the budget guard coordinate through
  local file locks and in-process state. Multi-replica deployments need a shared store; a network
  filesystem is not one.
- **Tamper-proof audit.** The activation log is tamper-*evident*. Someone with write access to the
  data directory can rebuild the chain; publish the head hash elsewhere. The budget file and the
  activation store are hashed and locked, not signed.
- **User identity and key rotation.** One API-key scheme with four scopes. JWT/OIDC, expiry and
  rotation are not implemented.
- **Complete PII detection.** Redaction is pattern-based and misses names, postal addresses and
  free-form identifiers. Hashed n-gram features are not prompt text, but they are derived from it --
  treat the feature matrix as sensitive.
- **Live model evaluation.** Tests, demo and benches never call a paid API. `AnthropicProvider` is
  tested against a mocked client, and no cassettes ship.
- **An MLflow server.** Runs are logged to the local file backend; the policy registry is a local
  content-addressed store.

## License

MIT. Copyright (c) 2026 Rouviere German.
