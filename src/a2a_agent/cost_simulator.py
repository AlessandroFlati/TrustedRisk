"""Phase 17.AA - Cost simulator vs LACE-only baseline.

Runs a synthetic prospective cohort through both:

  - **LACE-only baseline**: threshold the calibrated risk at 0.20 ->
    discharge_home / continued_admission, no abstain, no fairness
    layer.
  - **TrustedRisk pipeline**: the deterministic floor of the 4-critic
    + 3-agent debate + patient-advocate stack (re-uses
    `prospective_eval`).

Then computes:
  - readmissions averted (per the calibrated probabilities)
  - hospital cost saved per averted readmission (HCUP anchor)
  - QALYs gained per averted readmission (literature anchor)
  - net cost savings, per payer perspective
    (Medicare / Medicaid / commercial / uninsured)

References:
- AHRQ HCUP Statistical Brief #248 (HRRP, 2019) - $14,400 mean
  inpatient readmission cost.
- Fingar et al. 2019 - HRRP cost-effectiveness.
- ICER per QALY thresholds: $50k / $100k / $150k.

Pure-Python deterministic. No NumPy.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from .prospective_eval import (
    DISPOSITION_THRESHOLD, _calibrated_p, _ci_width_for_lace,
    _compose_outcome_rate, _evaluate_encounter,
    _sample_demographics, _sample_lace,
)


# Per-payer literature cost anchors (USD)
_PAYER_READMIT_COST: dict[str, float] = {
    "medicare":   14_400.0,   # HCUP HRRP 2019
    "medicaid":   12_900.0,   # HCUP 2022 Medicaid
    "commercial": 16_200.0,   # HCUP 2022 commercial
    "uninsured":  11_500.0,   # HCUP 2022 self-pay
    "tricare":    14_400.0,
}

# QALY uplift per averted 30-day readmission (Fingar 2019 anchor)
_QALY_PER_AVERTED = 0.05
# WTP threshold per QALY (ICER value-based pricing)
_DEFAULT_WTP_PER_QALY = 100_000.0


# ─────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────


class CostBreakdown(BaseModel):
    n: int
    readmissions_predicted: int
    readmissions_averted_vs_baseline: int
    abstain_rate: float = Field(ge=0.0, le=1.0)
    cost_saved_usd: float
    qalys_gained: float
    icer_per_qaly: float
    incremental_net_monetary_benefit: float
    rationale: str


class CostSimulationReport(BaseModel):
    cohort_n: int
    overall: CostBreakdown
    by_payer: dict[str, CostBreakdown]
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Counterfactual outcome model
# ─────────────────────────────────────────────────────────────────────


def _expected_outcome(
    action: str, base_risk: float,
) -> float:
    """Expected 30-day readmission probability conditional on the
    deployed action. Anchors:
      - discharge_home: full predicted risk.
      - discharge_with_homecare: 0.85x (literature 10-20% reduction).
      - continued_admission: 0.55x (clinical-care intensification).
      - abstain: 1.0x (proxy = same as baseline).
    """
    multiplier = {
        "discharge_home": 1.0,
        "discharge_with_homecare": 0.85,
        "continued_admission": 0.55,
        "abstain": 1.0,
    }.get(action, 1.0)
    return min(1.0, max(0.0, base_risk * multiplier))


def _baseline_action(risk: float) -> str:
    return (
        "continued_admission"
        if risk >= DISPOSITION_THRESHOLD
        else "discharge_home"
    )


def _readmissions_for_arm(
    rows, *, arm: Literal["pipeline", "baseline"],
) -> tuple[int, int]:
    """Sum of expected readmissions + count of abstains for the arm."""
    pred_readmits = 0.0
    n_abstain = 0
    for r in rows:
        action = (
            r.final_action if arm == "pipeline"
            else _baseline_action(r.risk_point_estimate)
        )
        if action == "abstain":
            n_abstain += 1
        pred_readmits += _expected_outcome(
            action, r.risk_point_estimate)
    return int(round(pred_readmits)), n_abstain


def _by_payer_rows(rows):
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r.insurance_type, []).append(r)
    return by


def _cost_breakdown(
    rows, *, payer: str | None = None,
    wtp_per_qaly: float = _DEFAULT_WTP_PER_QALY,
) -> CostBreakdown:
    n = len(rows)
    if n == 0:
        return CostBreakdown(
            n=0, readmissions_predicted=0,
            readmissions_averted_vs_baseline=0,
            abstain_rate=0.0, cost_saved_usd=0.0,
            qalys_gained=0.0, icer_per_qaly=0.0,
            incremental_net_monetary_benefit=0.0,
            rationale="empty cohort slice",
        )
    pipeline_pred, pipeline_abst = _readmissions_for_arm(
        rows, arm="pipeline")
    baseline_pred, _ = _readmissions_for_arm(
        rows, arm="baseline")
    averted = max(0, baseline_pred - pipeline_pred)
    payer_lookup = (payer or rows[0].insurance_type)
    cost_per_readmit = _PAYER_READMIT_COST.get(
        payer_lookup, 14_400.0)
    cost_saved = averted * cost_per_readmit
    qalys = averted * _QALY_PER_AVERTED
    icer = (cost_saved / qalys) if qalys > 0 else 0.0
    inmb = (qalys * wtp_per_qaly) - 0.0   # no incremental cost in
                                          # the deterministic floor
    return CostBreakdown(
        n=n,
        readmissions_predicted=pipeline_pred,
        readmissions_averted_vs_baseline=averted,
        abstain_rate=pipeline_abst / n,
        cost_saved_usd=round(cost_saved, 2),
        qalys_gained=round(qalys, 4),
        icer_per_qaly=round(icer, 2),
        incremental_net_monetary_benefit=round(inmb, 2),
        rationale=(
            f"n={n}, pipeline predicted {pipeline_pred} readmits, "
            f"baseline predicted {baseline_pred}, "
            f"averted {averted} -> ${cost_saved:,.0f} saved, "
            f"{qalys:.2f} QALYs gained."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def simulate_cost_impact(
    *,
    n_encounters: int = 10_000,
    seed: int = 20260430,
    fairness_audit_present: bool = True,
    wtp_per_qaly: float = _DEFAULT_WTP_PER_QALY,
) -> CostSimulationReport:
    """Run the prospective cohort + emit the cost-impact report."""
    if n_encounters <= 0:
        raise ValueError("n_encounters must be > 0")
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_encounters):
        lace = _sample_lace(rng)
        dem = _sample_demographics(rng)
        base = _calibrated_p(lace)
        eff = _compose_outcome_rate(base, dem)
        outcome = 1 if rng.random() < eff else 0
        rows.append(_evaluate_encounter(
            enc_id=i, lace=lace, dem=dem,
            risk_estimate=base,
            ci_width=_ci_width_for_lace(lace),
            outcome=outcome,
            fairness_audit_present=fairness_audit_present,
        ))
    overall = _cost_breakdown(rows, wtp_per_qaly=wtp_per_qaly)
    payer_buckets = _by_payer_rows(rows)
    by_payer = {
        payer: _cost_breakdown(
            payer_rows, payer=payer, wtp_per_qaly=wtp_per_qaly)
        for payer, payer_rows in payer_buckets.items()
    }
    return CostSimulationReport(
        cohort_n=n_encounters,
        overall=overall, by_payer=by_payer,
        references=[
            "AHRQ HCUP Statistical Brief #248 (HRRP, 2019).",
            "Fingar et al. 2019 - HRRP cost-effectiveness "
            "(Medical Care Research & Review).",
            "ICER WTP threshold $100k / QALY (2019 update).",
        ],
    )
