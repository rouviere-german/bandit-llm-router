"""Building policies from settings, and rebuilding them from registry snapshots.

Two directions, both needed by the service:

* :func:`build_policy` constructs a fresh policy from configuration. The configuration ends up
  inside the policy version, so two policies that differ in any way that matters -- floor,
  exploration parameter, discount, feature version, arm set -- can never share a version string.
* :func:`policy_from_snapshot` rebuilds a policy from what the model registry stored (JSON
  parameters plus ``.npz`` posterior arrays, never pickle) and refuses a snapshot whose recomputed
  version does not match the version recorded in it. That check is what makes "this is the policy
  that was screened and approved" a verifiable statement rather than a label.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from bandit_llm_router.ml.serialization import loads_npz
from bandit_llm_router.policies.base import PolicySnapshot
from bandit_llm_router.policies.linear_bandit import (
    EpsilonGreedy,
    LinearBanditPolicy,
    LinearThompson,
    LinUCB,
)

__all__ = [
    "POLICY_KINDS",
    "PolicyHyperparameters",
    "build_policy",
    "policy_from_registry_files",
    "policy_from_snapshot",
]

FloatArray = npt.NDArray[np.float64]
POLICY_KINDS: Final[tuple[str, ...]] = ("linucb", "thompson", "epsilon_greedy")


@dataclass(frozen=True, slots=True)
class PolicyHyperparameters:
    """Everything configurable about a learning policy."""

    kind: str = "linucb"
    alpha: float = 0.5
    epsilon: float = 0.05
    posterior_sd: float = 0.3
    mc_samples: int = 24
    ridge: float = 1.0
    discount: float = 1.0
    propensity_floor: float = 0.02

    def __post_init__(self) -> None:
        if self.kind not in POLICY_KINDS:
            raise ValueError(f"unknown policy kind {self.kind!r}; expected one of {POLICY_KINDS}")


def build_policy(
    hyperparameters: PolicyHyperparameters,
    *,
    arms: tuple[str, ...],
    dim: int,
    feature_version: str,
) -> LinearBanditPolicy:
    """Construct a fresh learning policy."""
    common: dict[str, Any] = {
        "arms": arms,
        "dim": dim,
        "ridge": hyperparameters.ridge,
        "discount": hyperparameters.discount,
        "propensity_floor": hyperparameters.propensity_floor,
        "feature_version": feature_version,
    }
    if hyperparameters.kind == "linucb":
        return LinUCB(alpha=hyperparameters.alpha, **common)
    if hyperparameters.kind == "thompson":
        return LinearThompson(
            posterior_sd=hyperparameters.posterior_sd,
            mc_samples=hyperparameters.mc_samples,
            **common,
        )
    return EpsilonGreedy(epsilon=hyperparameters.epsilon, **common)


def _number(params: Mapping[str, JsonValue], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # A stored snapshot is data, not a call site: a wrong type there is a corrupt
        # record, so it is reported as ValueError like every other snapshot defect.
        raise ValueError(f"snapshot parameter {key!r} must be a number")  # noqa: TRY004
    return float(value)


def _hyperparameters_from_params(params: Mapping[str, JsonValue]) -> PolicyHyperparameters:
    name = str(params.get("policy_name", ""))
    if name not in POLICY_KINDS:
        raise ValueError(f"snapshot names unknown policy {name!r}")
    return PolicyHyperparameters(
        kind=name,
        alpha=_number(params, "alpha", 0.5),
        epsilon=_number(params, "epsilon", 0.05),
        posterior_sd=_number(params, "posterior_sd", 0.3),
        mc_samples=int(_number(params, "mc_samples", 24)),
        ridge=_number(params, "ridge", 1.0),
        discount=_number(params, "discount", 1.0),
        propensity_floor=_number(params, "propensity_floor", 0.02),
    )


def policy_from_snapshot(snapshot: PolicySnapshot) -> LinearBanditPolicy:
    """Rebuild the policy a snapshot describes, verifying its version."""
    params = dict(snapshot.params)
    stored_arms = params.get("arms")
    if not isinstance(stored_arms, list):
        # A stored snapshot is data, not a call site: a wrong type there is a corrupt record.
        raise ValueError("snapshot parameter 'arms' must be a list of arm names")  # noqa: TRY004
    arms = tuple(str(arm) for arm in stored_arms)
    if len(arms) < 2:
        raise ValueError("snapshot does not list at least two arms")
    dim = int(_number(params, "dim", 0))
    if dim < 1:
        raise ValueError("snapshot does not record a feature dimension")
    policy = build_policy(
        _hyperparameters_from_params(params),
        arms=arms,
        dim=dim,
        feature_version=str(params.get("feature_version", "")),
    )
    if policy.version != snapshot.version:
        raise ValueError(
            f"snapshot version {snapshot.version} does not match the rebuilt policy "
            f"{policy.version}: the stored parameters were changed"
        )
    policy.load_snapshot(snapshot)
    return policy


def policy_from_registry_files(files: Mapping[str, bytes]) -> LinearBanditPolicy:
    """Rebuild a policy from the two files the registry stores for it."""
    missing = {"params.json", "posterior.npz"} - set(files)
    if missing:
        raise ValueError(f"registry version is missing {sorted(missing)}")
    params = json.loads(files["params.json"].decode("utf-8"))
    if not isinstance(params, dict):
        raise ValueError("params.json must contain a JSON object")  # noqa: TRY004
    arrays = loads_npz(files["posterior.npz"])
    version = str(params.get("policy_version", ""))
    name = str(params.get("policy_name", ""))
    snapshot = PolicySnapshot(
        name=name,
        version=version,
        params=params,
        arrays={key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
    )
    return policy_from_snapshot(snapshot)
