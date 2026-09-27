"""Provider layer and spend controls kept from the shared baseline.

The router needs four things from this kit: a provider protocol with offline implementations
(``providers``), token pricing (``pricing``), persisted step/token/USD budgets for recorded live
runs (``budget``), and the hash-chained audit log that policy activation writes to (``audit``). The
tool registry, envelopes and approval gate of the full agentic kit are not used here -- this service
calls one model per request and takes no tool actions.
"""
