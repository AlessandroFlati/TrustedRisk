"""healthcare.compute_quality_measures_aggregate /
compute_stars_rating_forecast / compute_care_gap_priority_ranking
-- Phase 10.1 HEDIS / CMS Stars Rating quality specialist (STARS-1/2/3).

Pain-point: CMS Medicare Stars Rating directly drives ~ $4-6 B/yr in
Quality Bonus Payments to Medicare Advantage plans. A single Star-
rating tier (e.g. 3.5 -> 4.0) can move a plan tens of millions in
QBP. The HEDIS measures that feed Stars are eligible-population +
numerator + denominator computations on a FHIR cohort.

Pure-deterministic (no LLM): the eligibility + numerator logic is
explicit per measure; LLM polish is reserved for the rationale text.

References:
- NCQA HEDIS Volume 2: Technical Specifications (2024)
- CMS 2025 Star Ratings Technical Notes
- Avalere CMS QBP analysis (2024)
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    CareGapPriorityAction,
    CareGapPriorityRanking,
    HEDISMeasureId,
    QualityMeasureRate,
    QualityMeasuresAggregateReport,
    StarsDomainId,
    StarsRatingForecast,
    StarsRatingForecastReport,
)


# ─────────────────────────────────────────────────────────────────────
# CMS Stars benchmarks (FY 2025)
# ─────────────────────────────────────────────────────────────────────

# (measure_id, name, weight, 5_star_cut, 4_star_cut, 3_star_cut, domain)
_MEASURE_TABLE: dict[str, dict[str, Any]] = {
    "BCS": {
        "name": "Breast Cancer Screening (50-74 women)",
        "weight": 1, "cuts": (0.79, 0.74, 0.69),
        "domain": "preventive_care",
    },
    "CCS": {
        "name": "Cervical Cancer Screening",
        "weight": 1, "cuts": (0.78, 0.74, 0.70),
        "domain": "preventive_care",
    },
    "COL": {
        "name": "Colorectal Cancer Screening (50-75)",
        "weight": 1, "cuts": (0.81, 0.75, 0.69),
        "domain": "preventive_care",
    },
    "CDC-EYE": {
        "name": "Eye Exam in Diabetes",
        "weight": 1, "cuts": (0.80, 0.73, 0.66),
        "domain": "chronic_conditions",
    },
    "CDC-HBA1C": {
        "name": "HbA1c Poor Control >9% (lower is better -- inverted)",
        "weight": 3, "cuts": (0.13, 0.18, 0.25), "inverted": True,
        "domain": "chronic_conditions",
    },
    "CBP": {
        "name": "Controlling High Blood Pressure (<140/90)",
        "weight": 3, "cuts": (0.79, 0.72, 0.65),
        "domain": "chronic_conditions",
    },
    "MPM-ACE": {
        "name": "Annual Monitoring for Persistent ACE/ARB",
        "weight": 1, "cuts": (0.93, 0.91, 0.89),
        "domain": "chronic_conditions",
    },
    "MPM-DIURETIC": {
        "name": "Annual Monitoring for Persistent Diuretics",
        "weight": 1, "cuts": (0.93, 0.91, 0.89),
        "domain": "chronic_conditions",
    },
    "MPM-ANTICONV": {
        "name": "Annual Monitoring for Persistent Anticonvulsants",
        "weight": 1, "cuts": (0.83, 0.78, 0.73),
        "domain": "chronic_conditions",
    },
    "OMW": {
        "name": "Osteoporosis Mgmt in Women w/ Fracture",
        "weight": 1, "cuts": (0.65, 0.55, 0.45),
        "domain": "chronic_conditions",
    },
    "PCR": {
        "name": "Plan All-Cause Readmissions (lower is better -- inverted)",
        "weight": 3, "cuts": (0.07, 0.10, 0.13), "inverted": True,
        "domain": "chronic_conditions",
    },
    "FUH": {
        "name": "Follow-Up After Mental Health Hosp (30-day)",
        "weight": 1, "cuts": (0.65, 0.55, 0.45),
        "domain": "chronic_conditions",
    },
    "FUM": {
        "name": "Follow-Up After ED for Mental Health (30-day)",
        "weight": 1, "cuts": (0.62, 0.50, 0.38),
        "domain": "chronic_conditions",
    },
    "AAB": {
        "name": "Avoidance of Antibiotic Treatment for Adults w/ Acute Bronchitis",
        "weight": 1, "cuts": (0.40, 0.35, 0.30), "inverted": True,
        "domain": "preventive_care",
    },
    "FMC": {
        "name": "Follow-Up After ED for Multiple Chronic Conditions",
        "weight": 1, "cuts": (0.65, 0.55, 0.45),
        "domain": "chronic_conditions",
    },
    "MRP": {
        "name": "Medication Reconciliation Post-Discharge",
        "weight": 1, "cuts": (0.85, 0.75, 0.65),
        "domain": "chronic_conditions",
    },
    "CWP": {
        "name": "Appropriate Testing for Pharyngitis",
        "weight": 1, "cuts": (0.85, 0.78, 0.71),
        "domain": "preventive_care",
    },
    "DAE": {
        "name": "Use of High-Risk Medications in the Elderly (lower better)",
        "weight": 1, "cuts": (0.10, 0.15, 0.20), "inverted": True,
        "domain": "chronic_conditions",
    },
    "TRC": {
        "name": "Transitions of Care (composite)",
        "weight": 1, "cuts": (0.78, 0.70, 0.62),
        "domain": "chronic_conditions",
    },
    "SUPD": {
        "name": "Statin Use in Persons with Diabetes",
        "weight": 1, "cuts": (0.86, 0.82, 0.78),
        "domain": "chronic_conditions",
    },
}


def _stars_for_rate(measure_id: str, rate: float) -> int:
    spec = _MEASURE_TABLE[measure_id]
    cuts = spec["cuts"]
    inverted = spec.get("inverted", False)
    if inverted:
        # Lower = better
        if rate <= cuts[0]: return 5
        if rate <= cuts[1]: return 4
        if rate <= cuts[2]: return 3
        return 2
    if rate >= cuts[0]: return 5
    if rate >= cuts[1]: return 4
    if rate >= cuts[2]: return 3
    return 2


def _coerce_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


async def compute_quality_measures_aggregate(
    measurement_year: int,
    cohort_summary: dict[str, dict[str, int]],
    n_eligible_patients: int | None = None,
) -> QualityMeasuresAggregateReport:
    """Aggregate per-measure HEDIS rates for a cohort.

    Args:
        measurement_year: e.g. 2025.
        cohort_summary: dict {measure_id: {numerator, denominator}}.
            Missing measures are skipped.
        n_eligible_patients: optional total cohort size.

    Returns:
        QualityMeasuresAggregateReport with per-measure rates + Stars.
    """
    measures: list[QualityMeasureRate] = []
    n_5 = n_4 = 0
    for mid, spec in _MEASURE_TABLE.items():
        if mid not in cohort_summary:
            continue
        n = int(cohort_summary[mid].get("numerator", 0))
        d = int(cohort_summary[mid].get("denominator", 0))
        rate = _coerce_rate(n, d)
        stars = _stars_for_rate(mid, rate)
        if stars == 5:
            n_5 += 1
        if stars >= 4:
            n_4 += 1
        eligible_unmet = max(0, d - n) if not spec.get("inverted") else n
        measures.append(QualityMeasureRate(
            measure_id=mid,                          # type: ignore[arg-type]
            measure_name=spec["name"],
            numerator=n, denominator=d, rate=rate,
            benchmark_5_star=spec["cuts"][0],
            benchmark_4_star=spec["cuts"][1],
            benchmark_3_star=spec["cuts"][2],
            current_stars=stars,
            eligible_unmet_count=eligible_unmet,
        ))

    rationale = (
        f"FY {measurement_year}: {len(measures)} measure(s) computed. "
        f"{n_5} measure(s) at 5-star, {n_4} at ≥ 4-star."
    )

    return QualityMeasuresAggregateReport(
        measurement_year=measurement_year,
        n_eligible_patients=n_eligible_patients or 0,
        measures=measures, n_measures=len(measures),
        overall_rate_5_star_count=n_5,
        overall_rate_4_star_count=n_4,
        rationale=rationale,
        references=[
            "NCQA HEDIS Volume 2: Technical Specifications (2024).",
            "CMS 2025 Star Ratings Technical Notes.",
        ],
    )


# Per-measure approximate "lift" from closing each unmet gap (not exact
# -- depends on cohort size; this is the per-patient marginal effect).
_PER_GAP_STARS_LIFT = 0.005


def _domain_average(rates: list[QualityMeasureRate],
                       domain: str) -> float:
    matching = [
        r.current_stars for r in rates
        if _MEASURE_TABLE[r.measure_id]["domain"] == domain
    ]
    if not matching:
        return 0.0
    return round(sum(matching) / len(matching), 2)


def _qbp_dollars(overall_stars: float) -> float:
    """Estimate Quality Bonus Payment dollars at the projected overall
    Stars rating. Per-plan benchmark; this returns a per-1k-member
    estimate that scales linearly."""
    # Avalere 2024: 4.0 unlocks ~5% QBP, 4.5+ unlocks +5% additional.
    # For a 1k-member plan at $11k/member-year benchmark: 5% = $550k
    base_qbp = 550_000.0   # 1k members at 4.0+ Stars
    if overall_stars < 4.0:
        return 0.0
    if overall_stars < 4.5:
        return base_qbp
    return base_qbp * 2.0   # 4.5+ doubles the bonus


async def compute_stars_rating_forecast(
    aggregate: QualityMeasuresAggregateReport | dict,
    contract_id: str | None = None,
    contract_size_thousand_members: float = 1.0,
) -> StarsRatingForecastReport:
    """Project per-domain + overall Stars rating at measurement-period
    end + estimate the QBP dollars at that rating."""
    if isinstance(aggregate, dict):
        aggregate = QualityMeasuresAggregateReport.model_validate(aggregate)

    # Per-domain current score = average per-measure star
    domains_seen = sorted({
        _MEASURE_TABLE[r.measure_id]["domain"] for r in aggregate.measures
    })
    forecasts: list[StarsRatingForecast] = []
    for d in domains_seen:
        rates = aggregate.measures
        cur = _domain_average(rates, d)
        # Project: assume the measures with highest unmet count close
        # 30 % of their gaps by EOM
        domain_measures = [
            r for r in rates if _MEASURE_TABLE[r.measure_id]["domain"] == d
        ]
        avg_lift = sum(
            _PER_GAP_STARS_LIFT * 0.30 * r.eligible_unmet_count
            for r in domain_measures
        ) / max(len(domain_measures), 1)
        proj = round(min(5.0, cur + avg_lift), 2)
        actions_4 = max(0, sum(
            (r.eligible_unmet_count if r.current_stars < 4 else 0)
            for r in domain_measures
        ))
        actions_5 = max(0, sum(
            (r.eligible_unmet_count if r.current_stars < 5 else 0)
            for r in domain_measures
        ))
        forecasts.append(StarsRatingForecast(
            domain=d,                                # type: ignore[arg-type]
            current_score=cur, projected_score_eom=proj,
            projection_basis=(
                f"Per-domain mean Stars + 30 % gap-closure lift "
                f"({avg_lift:+.3f} domain points)"
            ),
            actions_required_for_4_star=actions_4,
            actions_required_for_5_star=actions_5,
        ))

    cur_overall = round(
        sum(f.current_score for f in forecasts)
        / max(len(forecasts), 1),
        2,
    )
    proj_overall = round(
        sum(f.projected_score_eom for f in forecasts)
        / max(len(forecasts), 1),
        2,
    )
    qbp = _qbp_dollars(proj_overall) * contract_size_thousand_members

    rationale = (
        f"FY {aggregate.measurement_year} Stars forecast -- current "
        f"{cur_overall:.2f}, projected EOM {proj_overall:.2f}. "
        f"Projected QBP at projected overall: $"
        f"{qbp:,.0f} (per 1k members × {contract_size_thousand_members:.1f}k)."
    )

    return StarsRatingForecastReport(
        contract_id=contract_id,
        measurement_year=aggregate.measurement_year,
        domains=forecasts,
        overall_current=cur_overall,
        overall_projected_eom=proj_overall,
        estimated_qbp_dollars_at_overall=round(qbp, 2),
        rationale=rationale,
        references=[
            "CMS 2025 Star Ratings Technical Notes.",
            "Avalere Health: CMS QBP Analysis (2024).",
        ],
    )


# Per-measure marginal cost-of-closure (1 = easy, 5 = very hard).
_CLOSURE_DIFFICULTY: dict[str, str] = {
    "BCS":         "easy",
    "CCS":         "easy",
    "COL":         "moderate",
    "CDC-EYE":     "moderate",
    "CDC-HBA1C":   "very_hard",
    "CBP":         "hard",
    "MPM-ACE":     "easy",
    "MPM-DIURETIC": "easy",
    "MPM-ANTICONV": "easy",
    "OMW":         "moderate",
    "PCR":         "very_hard",
    "FUH":         "hard",
    "FUM":         "hard",
    "AAB":         "moderate",
    "FMC":         "hard",
    "MRP":         "moderate",
    "CWP":         "easy",
    "DAE":         "moderate",
    "TRC":         "moderate",
    "SUPD":        "easy",
}

_CLOSURE_INTERVENTIONS: dict[str, str] = {
    "BCS":         "Mammography reminder + scheduling outreach.",
    "CCS":         "Pap test reminder; cytology lab order at next visit.",
    "COL":         "FIT kit mailing + colonoscopy referral pathway.",
    "CDC-EYE":     "Tele-retinal screening kiosk at PCP visits.",
    "CDC-HBA1C":   "Pharmacist-led MTM + insulin titration protocol.",
    "CBP":         "Home BP monitoring + nurse-managed titration.",
    "MPM-ACE":     "Lab-order auto-renewal at 11-month mark.",
    "MPM-DIURETIC": "Lab-order auto-renewal at 11-month mark.",
    "MPM-ANTICONV": "Lab-order auto-renewal at 11-month mark.",
    "OMW":         "DEXA + bisphosphonate referral pathway.",
    "PCR":         "Post-discharge transition-of-care bundle (Project RED).",
    "FUH":         "Inpatient psychiatric -> outpatient appt pre-scheduled.",
    "FUM":         "ED -> BH navigator within 7 days.",
    "AAB":         "Antibiotic stewardship clinical-decision-support alerts.",
    "FMC":         "ED -> PCP appointment within 14 days, navigator-led.",
    "MRP":         "Post-discharge pharmacist call within 30 days.",
    "CWP":         "Strep test before antibiotic Rx; CDS pop-up.",
    "DAE":         "Beers-criteria pharmacist deprescribing review.",
    "TRC":         "Composite -- improve every component (notification, "
                       "med recon, PCP follow-up).",
    "SUPD":        "Statin auto-prescribe at PCP visit for diabetics 40-75.",
}


async def compute_care_gap_priority_ranking(
    aggregate: QualityMeasuresAggregateReport | dict,
    *,
    contract_size_thousand_members: float = 1.0,
    top_n: int = 10,
) -> CareGapPriorityRanking:
    """Rank care-gap-closure actions by Stars-impact / difficulty / QBP.

    Returns the top-N actions (default 10) with expected lift.
    """
    if isinstance(aggregate, dict):
        aggregate = QualityMeasuresAggregateReport.model_validate(aggregate)

    actions: list[CareGapPriorityAction] = []
    for r in aggregate.measures:
        if r.eligible_unmet_count == 0:
            continue
        spec = _MEASURE_TABLE[r.measure_id]
        weight = spec["weight"]
        # Lift = unmet × per-gap × weight × difficulty inverse (easy=1, hard=0.4)
        difficulty = _CLOSURE_DIFFICULTY[r.measure_id]
        diff_factor = {"easy": 1.0, "moderate": 0.7, "hard": 0.4,
                          "very_hard": 0.2}[difficulty]
        expected_stars = round(
            r.eligible_unmet_count * _PER_GAP_STARS_LIFT * weight * diff_factor,
            3,
        )
        # QBP impact: only when projected lifts cross 4.0 -> 4.5 cuts
        # We approximate as 0.5% of base QBP per Stars-lift unit
        expected_qbp = expected_stars * 110_000.0 * contract_size_thousand_members
        actions.append(CareGapPriorityAction(
            measure_id=r.measure_id,                 # type: ignore[arg-type]
            measure_name=r.measure_name,
            n_eligible_patients_with_gap=r.eligible_unmet_count,
            expected_stars_lift=expected_stars,
            expected_qbp_lift_dollars=round(expected_qbp, 2),
            closure_difficulty=difficulty,           # type: ignore[arg-type]
            suggested_intervention=_CLOSURE_INTERVENTIONS[r.measure_id],
        ))

    actions.sort(key=lambda a: -a.expected_qbp_lift_dollars)
    actions = actions[:top_n]

    cum_stars = round(sum(a.expected_stars_lift for a in actions), 3)
    cum_qbp = round(sum(a.expected_qbp_lift_dollars for a in actions), 2)
    rationale = (
        f"Top {len(actions)} of {len(aggregate.measures)} measures "
        f"ranked by expected QBP lift. Cumulative expected Stars lift "
        f"{cum_stars:+.3f} -> cumulative QBP $"
        f"{cum_qbp:,.0f}."
    )

    return CareGapPriorityRanking(
        actions=actions, n_actions=len(actions),
        cumulative_expected_stars_lift=cum_stars,
        cumulative_expected_qbp_dollars=cum_qbp,
        rationale=rationale,
        references=[
            "NCQA HEDIS Vol 2 (2024) -- measure specs.",
            "CMS 2025 Star Ratings Technical Notes.",
            "AHRQ + Project RED -- readmission-reduction evidence.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_quality_measures_aggregate)
    mcp.tool()(compute_stars_rating_forecast)
    mcp.tool()(compute_care_gap_priority_ranking)
