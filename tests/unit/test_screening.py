"""The screening gate: what a candidate must prove before anyone may propose activating it.

Each gate is tested by making exactly one of them fail while the others pass, because a gate that
only ever fires together with another one is not a gate. The last test is the important one for
governance: a report is bound by content hash to the log it was computed on, so an approval cannot
be moved onto different evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from bandit_llm_router.ope.estimators import LoggedBatch
from bandit_llm_router.ope.screening import (
    CostBound,
    EstimateRecord,
    ScreeningConfig,
    ScreeningReport,
    TargetPolicy,
    screen_candidate,
)

pytestmark = pytest.mark.unit

N_ARMS = 3
DIM = 3


class FixedPolicy:
    """A policy that plays the same distribution everywhere: the simplest honest candidate."""

    def __init__(self, name: str, probabilities: tuple[float, ...]) -> None:
        self.name = name
        self._row = np.asarray(probabilities, dtype=np.float64)

    @property
    def version(self) -> str:
        return f"{self.name}-" + "0" * 12

    def probabilities_matrix(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        del rng
        return np.tile(self._row, (features.shape[0], 1))


def make_batch(
    n: int = 3_000, *, seed: int = 0, good_arm: int = 2, floor: float = 0.2
) -> LoggedBatch:
    """A log where arm ``good_arm`` earns 0.8 and the others 0.4, logged near-uniformly."""
    rng = np.random.default_rng(seed)
    features = np.column_stack([np.ones(n), rng.uniform(-1.0, 1.0, size=(n, DIM - 1)) * 0.5])
    row = np.full(N_ARMS, floor)
    row[0] += 1.0 - floor * N_ARMS
    actions = rng.choice(N_ARMS, size=n, p=row)
    means = np.where(actions == good_arm, 0.8, 0.4)
    rewards = np.clip(means + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)
    return LoggedBatch(
        features=features,
        actions=actions.astype(np.int64),
        propensities=row[actions],
        rewards=rewards,
        n_arms=N_ARMS,
    )


def costs(batch: LoggedBatch, *, cheap: float = 4_000.0) -> tuple[np.ndarray, np.ndarray]:
    """Expected and ceiling cost matrices in micro-USD, rising with the arm index."""
    scale = np.array([1.0, 2.0, 5.0])
    expected = np.tile(cheap * scale, (batch.n, 1))
    return expected, expected * 2.0


def screen(
    candidate: TargetPolicy,
    incumbent: TargetPolicy,
    *,
    batch: LoggedBatch | None = None,
    config: ScreeningConfig | None = None,
) -> ScreeningReport:
    rows = batch if batch is not None else make_batch()
    expected, ceiling = costs(rows)
    return screen_candidate(
        rows,
        candidate=candidate,
        incumbent=incumbent,
        expected_cost_micros=expected,
        ceiling_cost_micros=ceiling,
        config=config
        if config is not None
        else ScreeningConfig(bootstrap=200, min_ess=300.0, max_exploration_usd=10_000.0),
        rng=np.random.default_rng(1),
    )


# --- the value gate -----------------------------------------------------------------------------


def test_a_genuinely_better_candidate_is_eligible() -> None:
    report = screen(
        FixedPolicy("good", (0.05, 0.05, 0.90)), FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    )
    assert report.eligible
    assert report.reasons == ()
    assert report.dr_lower_bound > report.incumbent_dr.value
    assert report.candidate_dr.value > report.incumbent_dr.value
    assert report.improvement > 0.0


def test_a_genuinely_worse_candidate_is_rejected() -> None:
    report = screen(
        FixedPolicy("bad", (0.90, 0.05, 0.05)), FixedPolicy("incumbent", (0.20, 0.20, 0.60))
    )
    assert not report.eligible
    assert "dr_lower_bound_does_not_beat_incumbent" in report.reasons


def test_the_gate_uses_the_lower_bound_so_noise_is_not_enough() -> None:
    """A candidate barely ahead of the incumbent must not pass on its point estimate alone."""
    batch = make_batch(n=400, seed=3)
    tie = FixedPolicy("tie", (0.33, 0.33, 0.34))
    report = screen(tie, FixedPolicy("incumbent", (0.33, 0.34, 0.33)), batch=batch)
    assert report.candidate_dr.value == pytest.approx(report.incumbent_dr.value, abs=0.02)
    assert report.dr_lower_bound < report.candidate_dr.value
    assert not report.eligible


def test_a_required_margin_can_be_demanded_on_top() -> None:
    candidate = FixedPolicy("good", (0.05, 0.05, 0.90))
    incumbent = FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    lenient = screen(candidate, incumbent)
    assert lenient.eligible
    strict = screen(
        candidate,
        incumbent,
        config=ScreeningConfig(
            bootstrap=200, min_ess=300.0, max_exploration_usd=10_000.0, min_improvement=0.9
        ),
    )
    assert not strict.eligible
    assert "dr_lower_bound_does_not_beat_incumbent" in strict.reasons


# --- the evidence gate ---------------------------------------------------------------------------


def test_a_candidate_the_log_cannot_support_is_rejected_for_lack_of_evidence() -> None:
    """Concentrating on rarely logged actions collapses ESS; screening says so explicitly."""
    batch = make_batch(n=3_000, floor=0.02)
    report = screen(
        FixedPolicy("narrow", (0.0, 0.0, 1.0)),
        FixedPolicy("incumbent", (0.9, 0.05, 0.05)),
        batch=batch,
        config=ScreeningConfig(bootstrap=100, min_ess=1_000.0, max_exploration_usd=10_000.0),
    )
    assert "effective_sample_size_below_minimum" in report.reasons
    assert report.ess < 1_000.0
    assert not report.eligible


def test_ess_is_reported_even_when_the_candidate_passes() -> None:
    report = screen(
        FixedPolicy("good", (0.05, 0.05, 0.90)), FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    )
    assert report.ess > 300.0
    assert report.n == 3_000
    assert report.min_propensity > 0.0


# --- the cost gate -------------------------------------------------------------------------------


def test_a_candidate_whose_worst_case_busts_the_budget_is_rejected() -> None:
    """However good it looks: exploration is priced, and the price is a hard gate."""
    candidate = FixedPolicy("good", (0.05, 0.05, 0.90))
    report = screen(
        candidate,
        FixedPolicy("incumbent", (0.60, 0.20, 0.20)),
        config=ScreeningConfig(bootstrap=200, min_ess=300.0, max_exploration_usd=1.0),
    )
    assert "worst_case_cost_exceeds_budget" in report.reasons
    assert not report.eligible
    assert report.cost.worst_case_horizon_usd > 1.0


def test_the_cost_bound_prices_the_most_expensive_reachable_arm() -> None:
    """Any arm with positive probability can be drawn, so the worst case charges the dearest one."""
    batch = make_batch(n=500)
    expected, ceiling = costs(batch)
    cheap_only = screen_candidate(
        batch,
        candidate=FixedPolicy("cheap", (1.0, 0.0, 0.0)),
        incumbent=FixedPolicy("incumbent", (0.6, 0.2, 0.2)),
        expected_cost_micros=expected,
        ceiling_cost_micros=ceiling,
        config=ScreeningConfig(bootstrap=0),
        rng=np.random.default_rng(2),
    )
    explores = screen_candidate(
        batch,
        candidate=FixedPolicy("explores", (0.98, 0.01, 0.01)),
        incumbent=FixedPolicy("incumbent", (0.6, 0.2, 0.2)),
        expected_cost_micros=expected,
        ceiling_cost_micros=ceiling,
        config=ScreeningConfig(bootstrap=0),
        rng=np.random.default_rng(2),
    )
    # Expected spend barely moves; the worst case jumps to the expensive arm's ceiling.
    assert explores.cost.expected_usd_per_request == pytest.approx(
        cheap_only.cost.expected_usd_per_request, rel=0.1
    )
    assert explores.cost.worst_case_usd_per_request == pytest.approx(
        cheap_only.cost.worst_case_usd_per_request * 5.0
    )
    assert explores.cost.worst_case_horizon_usd == pytest.approx(
        explores.cost.worst_case_usd_per_request * explores.config.horizon_requests
    )


# --- the record ----------------------------------------------------------------------------------


def test_a_report_is_bound_by_hash_to_its_candidate_and_its_log() -> None:
    candidate = FixedPolicy("good", (0.05, 0.05, 0.90))
    incumbent = FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    first = screen(candidate, incumbent, batch=make_batch(n=800, seed=5))
    again = screen(candidate, incumbent, batch=make_batch(n=800, seed=5))
    other_log = screen(candidate, incumbent, batch=make_batch(n=800, seed=6))
    assert first.logs_digest == again.logs_digest
    assert first.logs_digest != other_log.logs_digest
    # created_at differs, so the whole-report digests differ even for identical evidence; the log
    # digest is the part activation re-checks.
    assert first.digest() == first.model_copy().digest()
    assert first.digest() != other_log.digest()


def test_editing_a_stored_report_changes_its_digest() -> None:
    report = screen(
        FixedPolicy("good", (0.05, 0.05, 0.90)), FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    )
    tampered = report.model_copy(update={"dr_lower_bound": report.dr_lower_bound + 1.0})
    assert tampered.digest() != report.digest()


def test_the_summary_row_carries_the_numbers_a_reviewer_reads() -> None:
    report = screen(
        FixedPolicy("good", (0.05, 0.05, 0.90)), FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    )
    row = report.summary_row()
    assert row["eligible"] is True
    assert row["reasons"] == "-"
    assert set(row) == {
        "candidate",
        "dr",
        "dr_lower",
        "incumbent_dr",
        "ess",
        "worst_case_usd",
        "eligible",
        "reasons",
    }


def test_all_three_estimators_are_recorded_for_the_candidate() -> None:
    report = screen(
        FixedPolicy("good", (0.05, 0.05, 0.90)), FixedPolicy("incumbent", (0.60, 0.20, 0.20))
    )
    assert report.candidate_ips.method == "ips"
    assert report.candidate_snips.method == "snips"
    assert report.candidate_dr.method == "dr"
    assert report.candidate_dr.ci_low == pytest.approx(report.dr_lower_bound)


def test_an_estimate_record_copies_every_field() -> None:
    from bandit_llm_router.ope.estimators import Estimate

    estimate = Estimate(
        method="dr",
        value=0.5,
        std_error=0.02,
        ci_low=0.46,
        ci_high=0.54,
        n=100,
        ess=88.0,
        clipped_fraction=0.03,
    )
    record = EstimateRecord.of(estimate)
    assert record.value == 0.5
    assert record.ess == 88.0
    assert record.clipped_fraction == 0.03


def test_screening_validates_the_cost_matrices() -> None:
    batch = make_batch(n=100)
    expected, ceiling = costs(batch)
    candidate = FixedPolicy("good", (0.1, 0.1, 0.8))
    incumbent = FixedPolicy("incumbent", (0.6, 0.2, 0.2))
    for bad_expected, bad_ceiling, match in (
        (expected[:10], ceiling, "expected_cost_micros must have shape"),
        (expected, ceiling[:10], "ceiling_cost_micros must have shape"),
        (-expected, ceiling, "costs must be non-negative"),
    ):
        with pytest.raises(ValueError, match=match):
            screen_candidate(
                batch,
                candidate=candidate,
                incumbent=incumbent,
                expected_cost_micros=bad_expected,
                ceiling_cost_micros=bad_ceiling,
                config=ScreeningConfig(bootstrap=0),
            )


def test_screening_config_is_frozen_and_versioned() -> None:
    config = ScreeningConfig()
    assert config.digest_payload()["min_ess"] == config.min_ess
    with pytest.raises(ValueError, match="frozen"):
        config.min_ess = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ScreeningConfig(alpha=0.9)


def test_a_cost_bound_refuses_negative_money() -> None:
    CostBound(
        expected_usd_per_request=0.1, worst_case_usd_per_request=0.2, worst_case_horizon_usd=20.0
    )
    with pytest.raises(ValidationError):
        CostBound(
            expected_usd_per_request=-0.1,
            worst_case_usd_per_request=0.2,
            worst_case_horizon_usd=20.0,
        )
