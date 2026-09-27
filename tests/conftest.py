"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked.

The router fixtures build a *small* world on purpose: a 512-request workload pool and a compact
featurizer keep a whole simulated run inside a test's time budget while exercising exactly the code
paths the benches run at full size.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from hypothesis import HealthCheck, settings

from bandit_llm_router.arms import ArmCatalog, load_catalog
from bandit_llm_router.features import Featurizer
from bandit_llm_router.policies.linear_bandit import LinUCB
from bandit_llm_router.reward import RewardConfig
from bandit_llm_router.simulator import QualityShift, SimulatedEnvironment, Workload
from bandit_llm_router.simulator.workload import FAMILIES

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_PATH = REPO_ROOT / "config" / "pricing.toml"
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


# --- router fixtures --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def catalog() -> ArmCatalog:
    """The repo's real pricing config: tests price arms the way production does."""
    return load_catalog(PRICING_PATH)


@pytest.fixture(scope="session")
def featurizer() -> Featurizer:
    return Featurizer()


@pytest.fixture(scope="session")
def workload() -> Workload:
    return Workload(seed=11, pool_size=512)


@pytest.fixture(scope="session")
def environment(
    workload: Workload, catalog: ArmCatalog, featurizer: Featurizer
) -> SimulatedEnvironment:
    """A stationary simulated environment shared by the read-only tests."""
    return SimulatedEnvironment(
        workload=workload,
        catalog=catalog,
        reward_config=RewardConfig(),
        featurizer=featurizer,
    )


@pytest.fixture(scope="session")
def shifting_environment(
    workload: Workload, catalog: ArmCatalog, featurizer: Featurizer
) -> SimulatedEnvironment:
    """The same world with one mid-run quality shift: Opus degrades on ``code``."""
    return SimulatedEnvironment(
        workload=workload,
        catalog=catalog,
        featurizer=featurizer,
        shifts=(QualityShift(at_step=1_000, family="code", arm="opus", intercept_delta=-2.5),),
    )


@pytest.fixture(scope="session")
def severe_shift_environment(
    workload: Workload, catalog: ArmCatalog, featurizer: Featurizer
) -> SimulatedEnvironment:
    """A shift the router cannot shrug off: the arm it learned to prefer becomes the worst.

    ``shifting_environment`` degrades one arm on one family, which is realistic but too small to
    measure adaptation on a few thousand requests. Here Haiku -- best on about three quarters of the
    traffic before the shift -- degrades on every family, so the policy's learned preference becomes
    actively wrong and a stationary posterior has something to be slow about.
    """
    return SimulatedEnvironment(
        workload=workload,
        catalog=catalog,
        featurizer=featurizer,
        shifts=tuple(
            QualityShift(at_step=2_000, family=family, arm="haiku", intercept_delta=-3.5)
            for family in FAMILIES
        ),
    )


@pytest.fixture
def linucb(environment: SimulatedEnvironment) -> LinUCB:
    return LinUCB(
        arms=environment.arms,
        dim=environment.feature_dim,
        feature_version=environment.featurizer.version,
        propensity_floor=0.05,
        alpha=0.5,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
