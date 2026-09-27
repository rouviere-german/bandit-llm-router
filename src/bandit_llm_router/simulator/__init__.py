"""Offline simulator: seeded workload, per-arm quality curves and the array-shaped environment.

Nothing here calls a provider. The simulator exists so that learning, off-policy evaluation and
policy screening can be exercised end to end -- including scheduled quality shifts and adversarial
feedback -- without an API key and without spending money.
"""

from bandit_llm_router.simulator.environment import Outcome, Regime, SimulatedEnvironment
from bandit_llm_router.simulator.quality import (
    DEFAULT_ARM_QUALITY,
    ArmQualitySpec,
    QualityModel,
    QualityShift,
)
from bandit_llm_router.simulator.workload import (
    FAMILIES,
    FAMILY_SPECS,
    FamilySpec,
    SyntheticRequest,
    Workload,
)

__all__ = [
    "DEFAULT_ARM_QUALITY",
    "FAMILIES",
    "FAMILY_SPECS",
    "ArmQualitySpec",
    "FamilySpec",
    "Outcome",
    "QualityModel",
    "QualityShift",
    "Regime",
    "SimulatedEnvironment",
    "SyntheticRequest",
    "Workload",
]
