# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | all four benches (baseline, regret, OPE, router) against `bench/thresholds.json` |
| `demo` | the offline demo (no API key) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for properties
  (invariants, round trips, oracles) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and bench must not touch the network or need
  `ANTHROPIC_API_KEY`; `tests/conftest.py` blocks non-loopback sockets. Use `FakeProvider` or
  `ReplayProvider` for model behavior.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them -- in practice, from the committed `bench/results/*.json`, so
  the reader can check the figure against the file. A plain bench run writes to the gitignored
  `bench/results/latest/` and leaves the tree clean; `--update-reference` rewrites the committed copy,
  which is a deliberate act and should land in the same commit as the doc change it justifies.
- **Threat model.** New entry points, tools or artifacts need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **The logged propensity is sacred.** Any change to the serving path that alters the probability
  with which an arm is chosen must alter the logged propensity in the same commit. Restriction by the
  breaker or the budget guard logs the *realized* distribution (ADR 0005); logging the policy's
  intended distribution instead biases every off-policy estimate and every promotion built on it,
  permanently and silently. If you are unsure whether a change does this, it needs a test that
  follows a decision into the log and back out through an estimator.
- **Bench metrics are a safety net, so treat their arithmetic as production code.** A threshold that
  passes for the wrong reason is worse than no threshold. Metrics that must stay *high* are recorded
  as shortfalls so every comparison in `bench/_common.py` runs one direction, and sentinel values
  need a test (`tests/unit/test_bench_thresholds.py`). A comparison between two configurations also
  needs a threshold that fails when the two stop differing -- `empty_rejection_sweep`,
  `trough_shortfall_stationary`, `online_log_barely_moved` and `online_path_not_weaker_than_full` all
  exist because the interesting number would otherwise pass vacuously.
- **Measure the path that ships, not a stronger one.** If a defense or a gate runs in a reduced form
  on the live path, the bench publishes *that* form as its own row rather than folding it into the
  stronger number: the online posterior's poisoning row and the adaptive-log coverage table are both
  there for this reason. A benched configuration that no deployed code path takes is a claim about
  nothing.
- **No pickle.** Use JSON, npz with `allow_pickle=False`, safetensors or faiss `write_index`.
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
