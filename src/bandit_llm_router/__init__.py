"""Contextual-bandit routing across Claude models, with logged propensities and OPE-gated rollout.

Layout:

- ``core``: settings, redacting JSON logs, tracing, metrics, the hardened FastAPI app factory.
- ``agentic``: the LLM provider protocol (Fake/Replay/Anthropic), token pricing, persisted budgets
  and the hash-chained audit log the activation gate writes to.
- ``ml``: the decision-log data contract, run manifests, the content-addressed model registry, PSI
  and KS drift monitors, and pickle-free serialization.
- ``arms`` / ``features`` / ``reward``: what can be routed to, what a request looks like
numerically,
  and what "better" means.
- ``policies``: LinUCB, Thompson sampling, epsilon-greedy and the static baselines, all returning
  distributions with propensity floors.
- ``ope``: IPS, SNIPS and doubly robust estimators, and the screening gate in front of activation.
- ``simulator`` / ``experiments``: the offline workload, quality model and the experiments that
  produce every number in the README.
- ``service`` / ``activation`` / ``budgets`` / ``breaker`` / ``feedback`` / ``logstore``: the live
  path.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
