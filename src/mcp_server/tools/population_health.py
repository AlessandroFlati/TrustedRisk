"""healthcare.compute_syndromic_surveillance /
compute_vaccine_reminder_cohort / compute_outbreak_heatmap
-- Phase 10.2 population-health + outbreak detection
(POPHEALTH-1/2/3).

Pain-point: public-health departments + ACOs need (a) early detection
of syndromic clusters (CDC ESSENCE / NSSP), (b) vaccine-reminder
cohorts grouped by overdue cadence, (c) DP-noised outbreak heatmaps
that can be published without re-identification risk.

Pure-deterministic floor: Poisson z-score for clusters; CDC ACIP-
schedule deltas for vaccine cohorts; Laplace mechanism for the DP
heatmap (epsilon configurable).

References:
- Heffernan R et al. Syndromic surveillance in public health practice,
  New York City. Emerg Infect Dis. 2004; 10(5): 858–864.
- CDC NSSP (National Syndromic Surveillance Program) -- ESSENCE
  detection algorithm.
- ACIP Recommended Adult / Pediatric Immunization Schedules (2025).
- Dwork C, McSherry F, Nissim K, Smith A. Calibrating noise to
  sensitivity in private data analysis. TCC 2006.
"""

from __future__ import annotations

import math
import random
from typing import Any

from shared.schemas import (
    OutbreakHeatmapCell,
    OutbreakHeatmapReport,
    SyndromicCluster,
    SyndromicSurveillanceReport,
    VaccineReminderCohort,
    VaccineReminderCohortReport,
)


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-1 -- compute_syndromic_surveillance
# ─────────────────────────────────────────────────────────────────────


def _severity_from_z(z: float) -> str:
    if z >= 5.0:
        return "outbreak"
    if z >= 3.0:
        return "alert"
    if z >= 2.0:
        return "watch"
    return "informational"


async def compute_syndromic_surveillance(
    surveillance_period_start_iso: str,
    surveillance_period_end_iso: str,
    observed_counts_by_syndrome: dict[str, int],
    expected_counts_by_syndrome: dict[str, float],
    *,
    geographic_window: str | None = None,
    z_threshold: float = 2.0,
) -> SyndromicSurveillanceReport:
    """Detect clusters whose observed-vs-expected count delta exceeds
    a Poisson z-score threshold.

    Args:
        surveillance_period_start_iso: ISO-8601 inclusive start.
        surveillance_period_end_iso: ISO-8601 inclusive end.
        observed_counts_by_syndrome: dict of syndrome -> observed count.
        expected_counts_by_syndrome: dict of syndrome -> expected count
            (e.g. seasonal mean from a 5-year baseline).
        geographic_window: optional textual region (e.g. "NYC ZIP 10001").
        z_threshold: Poisson z-score cut for cluster admission
            (default 2.0 ≈ NSSP ESSENCE moderate-sensitivity).

    Returns:
        SyndromicSurveillanceReport with one row per detected cluster.
    """
    clusters: list[SyndromicCluster] = []
    n_total = sum(observed_counts_by_syndrome.values())

    for idx, (syndrome, observed) in enumerate(
        sorted(observed_counts_by_syndrome.items()), start=1,
    ):
        expected = expected_counts_by_syndrome.get(syndrome, 0.0)
        if expected <= 0.0:
            # Cannot compute z; skip -- public health workflow upstream
            # is responsible for backfilling baseline expectations.
            continue
        # Poisson approx: z = (O - E) / sqrt(E)
        z = (observed - expected) / math.sqrt(expected)
        if z < z_threshold:
            continue
        clusters.append(SyndromicCluster(
            cluster_id=f"CLUSTER-{idx:03d}",
            syndrome=syndrome,
            n_cases_observed=int(observed),
            n_cases_expected=round(expected, 2),
            z_score=round(z, 3),
            geographic_window=geographic_window,
            time_window_start_iso=surveillance_period_start_iso,
            time_window_end_iso=surveillance_period_end_iso,
            severity=_severity_from_z(z),                 # type: ignore[arg-type]
        ))

    n_outbreak = sum(1 for c in clusters if c.severity == "outbreak")
    rationale = (
        f"{len(clusters)} cluster(s) above z = {z_threshold:.2f} between "
        f"{surveillance_period_start_iso} and {surveillance_period_end_iso}. "
        f"{n_outbreak} at outbreak severity."
    )

    return SyndromicSurveillanceReport(
        surveillance_period_start_iso=surveillance_period_start_iso,
        surveillance_period_end_iso=surveillance_period_end_iso,
        n_total_chief_complaints=int(n_total),
        clusters=clusters,
        n_clusters=len(clusters),
        n_outbreak_severity=n_outbreak,
        rationale=rationale,
        references=[
            "Heffernan R et al. Syndromic surveillance in public health "
            "practice, NYC. Emerg Infect Dis. 2004; 10(5): 858-864.",
            "CDC NSSP / ESSENCE -- Poisson detection algorithm.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-2 -- compute_vaccine_reminder_cohort
# ─────────────────────────────────────────────────────────────────────

_OUTREACH_UPTAKE: dict[str, float] = {
    "phone":      0.30,
    "postcard":   0.18,
    "ehr_portal": 0.25,
    "sms":        0.22,
    "email":      0.15,
}


# Curated subset from ACIP 2025: vaccines + the trigger criteria for
# building a reminder cohort. The numerator is built UPSTREAM (e.g. via
# a HEDIS care-gap query); this tool just packages the cohort with
# per-channel uptake estimates and per-vaccine literature priors.
_VACCINE_TABLE: dict[str, dict[str, str]] = {
    "FLU":      {"name": "Influenza (annual)"},
    "PNEUMO":   {"name": "Pneumococcal (PCV20 / PPSV23 ≥ 65)"},
    "ZOSTER":   {"name": "Zoster (Shingrix ≥ 50)"},
    "TDAP":     {"name": "Tdap (every 10 yrs)"},
    "HPV":      {"name": "HPV (catch-up 9-26)"},
    "COVID-19": {"name": "COVID-19 booster (annual)"},
    "RSV":      {"name": "RSV (≥ 60, 1-time)"},
    "MMR":      {"name": "MMR (childhood + immune-comp boosters)"},
    "VARICELLA":{"name": "Varicella (childhood + immune-comp boosters)"},
    "MENACWY":  {"name": "MenACWY (adolescent)"},
}


async def compute_vaccine_reminder_cohort(
    overdue_by_vaccine: dict[str, int],
    *,
    age_distribution: dict[str, dict[str, int]] | None = None,
    risk_distribution: dict[str, dict[str, int]] | None = None,
    outreach_channel: str = "phone",
) -> VaccineReminderCohortReport:
    """Build per-vaccine reminder cohorts with per-channel uptake.

    Args:
        overdue_by_vaccine: dict {vaccine_id -> n_patients_overdue}.
        age_distribution: optional {vaccine_id -> {age_band -> n}}.
        risk_distribution: optional {vaccine_id -> {risk_label -> n}}.
        outreach_channel: one of {phone, postcard, ehr_portal, sms, email}.

    Returns:
        VaccineReminderCohortReport.
    """
    uptake = _OUTREACH_UPTAKE.get(outreach_channel, 0.20)
    age_distribution = age_distribution or {}
    risk_distribution = risk_distribution or {}

    cohorts: list[VaccineReminderCohort] = []
    total = 0
    for vid, n_overdue in sorted(overdue_by_vaccine.items()):
        spec = _VACCINE_TABLE.get(vid)
        if spec is None or n_overdue <= 0:
            continue
        cohorts.append(VaccineReminderCohort(
            vaccine_id=vid,
            vaccine_name=spec["name"],
            n_patients_overdue=int(n_overdue),
            age_distribution=age_distribution.get(vid, {}),
            risk_distribution=risk_distribution.get(vid, {}),
            expected_outreach_uptake=uptake,
        ))
        total += int(n_overdue)

    expected_uptake = round(uptake * total)
    rationale = (
        f"{len(cohorts)} vaccine cohort(s) covering {total} overdue "
        f"patient(s); expected uptake via {outreach_channel} ≈ "
        f"{expected_uptake} patient(s) at {uptake * 100:.0f}%."
    )

    return VaccineReminderCohortReport(
        cohorts=cohorts, n_cohorts=len(cohorts), total_patients=total,
        rationale=rationale,
        references=[
            "ACIP 2025 Recommended Adult Immunization Schedule.",
            "ACIP 2025 Recommended Childhood Immunization Schedule.",
            "CDC: Adult Vaccination Coverage NHIS 2023.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# POPHEALTH-3 -- compute_outbreak_heatmap (DP-noised)
# ─────────────────────────────────────────────────────────────────────


def _laplace_noise(scale: float, rng: random.Random) -> float:
    """Sample one Laplace(0, b) draw via inverse-CDF on a uniform."""
    u = rng.random() - 0.5
    sign = 1.0 if u >= 0 else -1.0
    return -sign * scale * math.log(1.0 - 2.0 * abs(u))


async def compute_outbreak_heatmap(
    counts_by_geo_syndrome: dict[str, dict[str, int]],
    population_by_geo: dict[str, int],
    *,
    epsilon: float = 1.0,
    seed: int = 4242,
) -> OutbreakHeatmapReport:
    """DP-noised outbreak heatmap suitable for public release.

    Args:
        counts_by_geo_syndrome: {geographic_bucket -> {syndrome -> count}}.
        population_by_geo: {geographic_bucket -> population}, used to
            compute per-100k rates after noising.
        epsilon: DP privacy budget (lower = more noise). Sensitivity
            assumed to be 1 (unit count contribution per individual).
        seed: RNG seed for the Laplace draws.

    Returns:
        OutbreakHeatmapReport (negative noised counts clipped to 0).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    rng = random.Random(seed)
    scale = 1.0 / epsilon
    cells: list[OutbreakHeatmapCell] = []

    for geo in sorted(counts_by_geo_syndrome):
        pop = max(1, population_by_geo.get(geo, 1))
        for syndrome, raw in sorted(counts_by_geo_syndrome[geo].items()):
            noised = max(0, int(round(raw + _laplace_noise(scale, rng))))
            rate = round(noised / pop * 100_000.0, 2)
            cells.append(OutbreakHeatmapCell(
                geographic_bucket=geo,
                syndrome=syndrome,
                n_cases_dp_noised=noised,
                rate_dp_noised=rate,
            ))

    rationale = (
        f"DP heatmap with {len(cells)} cell(s); Laplace mechanism, "
        f"ε = {epsilon:.2f}, sensitivity = 1 (per-individual count). "
        f"Negative noised counts clipped to 0; rates per 100k population."
    )

    return OutbreakHeatmapReport(
        cells=cells, n_cells=len(cells), epsilon_used=epsilon,
        rationale=rationale,
        references=[
            "Dwork C, McSherry F, Nissim K, Smith A. Calibrating noise "
            "to sensitivity in private data analysis. TCC 2006.",
            "CDC NSSP -- public-health surveillance disclosure controls.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_syndromic_surveillance)
    mcp.tool()(compute_vaccine_reminder_cohort)
    mcp.tool()(compute_outbreak_heatmap)
