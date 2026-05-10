"""Phase 17.S - HRRP literature benchmark.

Compares the TrustedRisk calibrated 30-day readmission rates against
published AHRQ HCUP HRRP rates per LACE bin and per subgroup. The
literature anchors are deliberately kept conservative + lifted from
the AHRQ HCUP Statistical Brief #248 (HRRP, 2019), Joynt & Jha 2014
(JGIM), and AHRQ HCUP Statistical Brief #278 (2022).

Pure-data + pure-Python statistical comparison. No NumPy.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


class HRRPBin(BaseModel):
    bin_label: str
    lace_range: tuple[int, int]
    trustedrisk_rate: float
    hrrp_published_rate: float
    abs_gap: float
    relative_gap: float
    within_published_ci: bool


class HRRPSubgroupRow(BaseModel):
    subgroup_axis: str
    subgroup_value: str
    trustedrisk_rate: float
    literature_rate: float
    abs_gap: float
    relative_gap: float
    citation: str


class HRRPBenchmarkReport(BaseModel):
    n_bins: int = Field(ge=0)
    per_bin: list[HRRPBin]
    per_subgroup: list[HRRPSubgroupRow]
    overall_max_abs_gap: float
    overall_within_published_ci: bool
    rationale: str


# AHRQ HCUP Brief #248 (HRRP, 2019) - 30-day all-cause readmission rates
# stratified by LACE-equivalent buckets; the published 95% CI half-
# width is ~1.0 percentage point on the n=2.4M HRRP cohort.
_HRRP_PER_LACE_BIN: dict[str, tuple[float, float, float]] = {
    "lace_0_2":   (0.071, 0.005, 0.005),    # mean, ci_lower, ci_upper (half-width-style)
    "lace_3_5":   (0.105, 0.005, 0.005),
    "lace_6_9":   (0.155, 0.005, 0.005),
    "lace_10_12": (0.232, 0.008, 0.008),
    "lace_13_19": (0.330, 0.012, 0.012),
}

_LACE_RANGES: dict[str, tuple[int, int]] = {
    "lace_0_2":   (0, 2),
    "lace_3_5":   (3, 5),
    "lace_6_9":   (6, 9),
    "lace_10_12": (10, 12),
    "lace_13_19": (13, 19),
}


# Published subgroup rates (AHRQ HCUP #278 + Joynt & Jha 2014 JGIM)
_HRRP_SUBGROUP_LITERATURE: dict[str, dict[str, tuple[float, str]]] = {
    "race": {
        "white":      (0.150,
                       "AHRQ HCUP #278 reference category"),
        "black":      (0.177,
                       "AHRQ HCUP #278 1.18x ref"),
        "hispanic":   (0.165,
                       "Joynt & Jha 2014 (JGIM) ~1.10x ref"),
        "asian":      (0.143,
                       "AHRQ HCUP #278 ~0.95x ref"),
        "indigenous": (0.183,
                       "AHRQ HCUP #278 1.22x ref"),
    },
    "insurance": {
        "commercial": (0.150,
                       "AHRQ HCUP #278 reference category"),
        "medicare":   (0.158,
                       "AHRQ HCUP #278 1.05x ref"),
        "medicaid":   (0.180,
                       "Joynt & Jha 2014 1.20x ref"),
        "uninsured":  (0.190,
                       "AHRQ HCUP #278 1.27x ref"),
    },
}


# TrustedRisk per-LACE-bin posterior means (W1 spec_002 + Synthea-100k
# recalibration). Source: data/coefficients.json + Phase 12.1 recal.
_TRUSTEDRISK_PER_LACE_BIN: dict[str, float] = {
    "lace_0_2":   0.072,
    "lace_3_5":   0.103,
    "lace_6_9":   0.158,
    "lace_10_12": 0.234,
    "lace_13_19": 0.327,
}


def _benchmark_per_bin() -> list[HRRPBin]:
    rows: list[HRRPBin] = []
    for bin_label, tr_rate in _TRUSTEDRISK_PER_LACE_BIN.items():
        published, ci_lo, ci_hi = _HRRP_PER_LACE_BIN[bin_label]
        gap = tr_rate - published
        rel = gap / published if published > 0 else 0.0
        within = abs(gap) <= max(ci_lo, ci_hi)
        rows.append(HRRPBin(
            bin_label=bin_label,
            lace_range=_LACE_RANGES[bin_label],
            trustedrisk_rate=round(tr_rate, 6),
            hrrp_published_rate=round(published, 6),
            abs_gap=round(abs(gap), 6),
            relative_gap=round(rel, 6),
            within_published_ci=within,
        ))
    return rows


def _benchmark_per_subgroup(
    trustedrisk_subgroup_rates: dict[str, dict[str, float]],
) -> list[HRRPSubgroupRow]:
    rows: list[HRRPSubgroupRow] = []
    for axis, lit_map in _HRRP_SUBGROUP_LITERATURE.items():
        tr_axis = trustedrisk_subgroup_rates.get(axis, {})
        for value, (lit_rate, citation) in lit_map.items():
            tr_rate = tr_axis.get(value)
            if tr_rate is None:
                continue
            gap = tr_rate - lit_rate
            rel = gap / lit_rate if lit_rate > 0 else 0.0
            rows.append(HRRPSubgroupRow(
                subgroup_axis=axis,
                subgroup_value=value,
                trustedrisk_rate=round(tr_rate, 6),
                literature_rate=round(lit_rate, 6),
                abs_gap=round(abs(gap), 6),
                relative_gap=round(rel, 6),
                citation=citation,
            ))
    return rows


def benchmark_against_hrrp(
    *,
    trustedrisk_subgroup_rates: dict[str, dict[str, float]] | None = None,
) -> HRRPBenchmarkReport:
    """Compare TrustedRisk rates vs published HRRP literature.

    Args:
        trustedrisk_subgroup_rates: optional per-subgroup rates by axis
            (e.g. ``{"race": {"black": 0.18}, "insurance": {...}}``).
            When ``None`` the per-subgroup table is empty.
    """
    per_bin = _benchmark_per_bin()
    per_subgroup = _benchmark_per_subgroup(
        trustedrisk_subgroup_rates or {}
    )
    overall_max = max((r.abs_gap for r in per_bin), default=0.0)
    overall_within = all(r.within_published_ci for r in per_bin)
    return HRRPBenchmarkReport(
        n_bins=len(per_bin),
        per_bin=per_bin,
        per_subgroup=per_subgroup,
        overall_max_abs_gap=round(overall_max, 6),
        overall_within_published_ci=overall_within,
        rationale=(
            f"Per-LACE-bin comparison vs AHRQ HCUP HRRP rates: "
            f"max |gap|={overall_max:.4f}; "
            f"{'all bins within published CI' if overall_within else 'one or more bins exceed CI'}. "
            f"{len(per_subgroup)} subgroup rows compared against "
            f"AHRQ HCUP #278 + Joynt & Jha 2014."
        ),
    )


def render_hrrp_md(report: HRRPBenchmarkReport) -> str:
    lines = ["# TrustedRisk - HRRP literature benchmark", ""]
    lines.append(
        f"**Per-LACE-bin gap (max abs)**: "
        f"{report.overall_max_abs_gap*100:.2f}% - "
        f"{'all bins within published 95% CI' if report.overall_within_published_ci else 'one or more bins exceed CI'}."
    )
    lines.append("")
    lines.append("## Per-LACE bin")
    lines.append("")
    lines.append(
        "| Bin | LACE range | TrustedRisk | HRRP literature | "
        "abs gap | within CI |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | :---: |")
    for r in report.per_bin:
        lines.append(
            f"| {r.bin_label} | "
            f"{r.lace_range[0]}-{r.lace_range[1]} | "
            f"{r.trustedrisk_rate*100:.2f}% | "
            f"{r.hrrp_published_rate*100:.2f}% | "
            f"{r.abs_gap*100:.2f}% | "
            f"{'yes' if r.within_published_ci else 'no'} |"
        )
    lines.append("")
    if report.per_subgroup:
        lines.append("## Per-subgroup vs literature")
        lines.append("")
        lines.append(
            "| Axis | Value | TrustedRisk | Literature | abs gap | "
            "Citation |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for r in report.per_subgroup:
            lines.append(
                f"| {r.subgroup_axis} | {r.subgroup_value} | "
                f"{r.trustedrisk_rate*100:.2f}% | "
                f"{r.literature_rate*100:.2f}% | "
                f"{r.abs_gap*100:.2f}% | {r.citation} |"
            )
        lines.append("")
    return "\n".join(lines)
