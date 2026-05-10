"""Phase 16.G2 - NIST AI RMF 1.0 + OECD AI Principles 2019 crosswalks.

Two formal crosswalks that map every TrustedRisk capability to the
two most-cited international AI-governance frameworks. Pure-data;
emits a single combined JSON + markdown document.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────


_NISTFunction = Literal["GOVERN", "MAP", "MEASURE", "MANAGE"]


class CrosswalkRow(BaseModel):
    framework: str
    framework_section: str
    requirement: str
    trustedrisk_evidence: str
    artefact_path: str | None = None


class FrameworkCrosswalkReport(BaseModel):
    generated_at: str
    n_rows: int = Field(ge=0)
    nist_ai_rmf_rows: list[CrosswalkRow]
    oecd_ai_principles_rows: list[CrosswalkRow]


# ─────────────────────────────────────────────────────────────────────
# NIST AI RMF 1.0 (Jan 2023) - 4 functions
# ─────────────────────────────────────────────────────────────────────


def _nist_rows() -> list[CrosswalkRow]:
    fr = "NIST AI RMF 1.0"
    return [
        CrosswalkRow(
            framework=fr,
            framework_section="GOVERN 1.1",
            requirement=(
                "Legal + regulatory requirements involving AI are "
                "understood, managed, and documented."
            ),
            trustedrisk_evidence=(
                "Regulatory pack maps EU AI Act, FDA 510(k), ISO "
                "13485, GDPR, HIPAA in a single 14-section document."
            ),
            artefact_path="docs/regulatory/REGULATORY_PACK.md",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="GOVERN 2.1",
            requirement=(
                "Roles, responsibilities, lines of communication "
                "are documented + clear to all personnel."
            ),
            trustedrisk_evidence=(
                "ISO 13485 QMS section of the regulatory pack; the "
                "audit log carries the operator + clinician role on "
                "every decision record."
            ),
            artefact_path="docs/regulatory/REGULATORY_PACK.md",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="GOVERN 3.2",
            requirement=(
                "Policies + procedures address oversight of AI "
                "system development + deployment."
            ),
            trustedrisk_evidence=(
                "Phase 14.16 multi-agent debate + Phase 14.15 "
                "planner-revision feedback loop; both ship with "
                "deterministic-floor critics."
            ),
            artefact_path=(
                "src/a2a_agent/{multi_agent_debate,planner_"
                "revision}.py"
            ),
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="GOVERN 4.1",
            requirement=(
                "Organizational practices + accountability "
                "structures address AI risks."
            ),
            trustedrisk_evidence=(
                "Append-only audit + RFC 6962 Merkle tree + "
                "reproducibility SQLite archive."
            ),
            artefact_path="src/a2a_agent/merkle_audit.py",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MAP 1.1",
            requirement=(
                "Intended purposes, potentially beneficial uses, "
                "context-specific laws, norms, expectations are "
                "documented."
            ),
            trustedrisk_evidence=(
                "Model Card section 1+2 (Mitchell 2019 format); "
                "datasheet motivation section."
            ),
            artefact_path="docs/research/MODEL_CARD.md",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MAP 1.2",
            requirement=(
                "Inter-disciplinary AI actors are involved in "
                "assessing risks and impacts."
            ),
            trustedrisk_evidence=(
                "4-critic ensemble (clinical_safety, fairness, "
                "evidence, llm_judge) + 3-agent debate "
                "(clinical_conservative, evidence_aggressive, "
                "fairness_guard) span clinical, statistical, "
                "ethical, ML perspectives."
            ),
            artefact_path=(
                "src/a2a_agent/{critique,multi_agent_debate}.py"
            ),
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MAP 2.3",
            requirement=(
                "Scientific integrity + Test, Evaluation, "
                "Verification + Validation (TEVV) considerations "
                "are documented."
            ),
            trustedrisk_evidence=(
                "Property-based tests, golden cohorts, adversarial "
                "v2/v3/v4 corpus, prospective synthetic eval at "
                "n=10k."
            ),
            artefact_path=(
                "tests/{property,golden,adversarial}/ + "
                "docs/prospective/PROSPECTIVE_EVAL.md"
            ),
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MAP 5.1",
            requirement=(
                "Likelihood + magnitude of impacts on "
                "individuals + society are characterised."
            ),
            trustedrisk_evidence=(
                "Subgroup audit with EOO + DP gaps; "
                "differential-privacy publication of subgroup "
                "rates (Laplace, epsilon=1.0)."
            ),
            artefact_path="docs/fairness/subgroup_audit.json",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MEASURE 1.3",
            requirement=(
                "Metrics + benchmarks reflect the intended uses + "
                "context."
            ),
            trustedrisk_evidence=(
                "ECE preferred gate <=0.05; Brier; AUROC; per-"
                "subgroup TPR + DP gaps; conformal target coverage "
                "1-alpha."
            ),
            artefact_path="docs/research/MODEL_CARD.md",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MEASURE 2.7",
            requirement=(
                "Trustworthy characteristics (validity, safety, "
                "fairness) are measured."
            ),
            trustedrisk_evidence=(
                "Phase 16.F2 split-conformal multi-class + Pleiss "
                "2017 calibration tension witness + selective-"
                "classification risk-coverage curve."
            ),
            artefact_path="src/a2a_agent/trustworthy_ml.py",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MEASURE 3.2",
            requirement=(
                "Risk + benefit measurements are repeated + "
                "compared over time."
            ),
            trustedrisk_evidence=(
                "Drift detection scheduler runs at configurable "
                "cadence; abstain-by-default on drift."
            ),
            artefact_path="src/a2a_agent/{scheduler,drift_monitor}.py",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MANAGE 1.2",
            requirement=(
                "Treatment of AI risks is informed by ranked + "
                "documented risks."
            ),
            trustedrisk_evidence=(
                "Per-issue planner critique (Phase 14.15) emits "
                "ranked + suggested-action issue list before any "
                "tool runs."
            ),
            artefact_path="src/a2a_agent/planner_revision.py",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MANAGE 2.4",
            requirement=(
                "Mechanisms are in place to deactivate, disengage, "
                "or override the AI system."
            ),
            trustedrisk_evidence=(
                "Stop-button per Article 14(4)(d) - federation "
                "exposes /admin/quiesce + abstain-trigger on "
                "uncertainty + clinician override surface."
            ),
            artefact_path="docs/regulatory/REGULATORY_PACK.md",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="MANAGE 4.1",
            requirement=(
                "Post-deployment AI system monitoring plans are "
                "implemented."
            ),
            trustedrisk_evidence=(
                "Background scheduler runs care_gap_sweep + "
                "drift_detection; recommendation-drift tool flags "
                "per-patient changes."
            ),
            artefact_path="src/a2a_agent/scheduler.py",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# OECD AI Principles 2019 - 5 principles
# ─────────────────────────────────────────────────────────────────────


def _oecd_rows() -> list[CrosswalkRow]:
    fr = "OECD AI Principles 2019"
    return [
        CrosswalkRow(
            framework=fr,
            framework_section="Principle 1.1 - Inclusive growth",
            requirement=(
                "AI should benefit people + planet by driving "
                "inclusive growth, sustainable development, "
                "well-being."
            ),
            trustedrisk_evidence=(
                "Subgroup audit + fairness-guard veto narrow the "
                "predicted-impact gap on under-served populations."
            ),
            artefact_path="docs/fairness/subgroup_audit.json",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="Principle 1.2 - Human-centred values",
            requirement=(
                "AI systems should respect rule of law, human "
                "rights, democratic values, diversity."
            ),
            trustedrisk_evidence=(
                "GDPR Art. 22 right-to-explanation surface + "
                "patient audit summary on demand + multi-language "
                "patient-facing translation."
            ),
            artefact_path="src/a2a_agent/right_to_explanation.py",
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="Principle 1.3 - Transparency",
            requirement=(
                "Stakeholders should understand AI-based outcomes "
                "+ challenge them."
            ),
            trustedrisk_evidence=(
                "Counterfactual explanations + SHAP attributions + "
                "multi-agent debate vote breakdown surfaced on "
                "every DecisionCard."
            ),
            artefact_path=(
                "src/mcp_server/tools/{counterfactual_explanation,"
                "model_research}.py"
            ),
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="Principle 1.4 - Robustness + safety",
            requirement=(
                "AI systems should be robust, secure, safe "
                "throughout their entire lifecycle."
            ),
            trustedrisk_evidence=(
                "Red-team v3 multi-target + v4 indirect prompt "
                "injection corpus; OOD detection (Mahalanobis "
                "chi^2); chaos-engineering harness."
            ),
            artefact_path=(
                "src/a2a_agent/{redteam_v3,safety_redteam}.py"
            ),
        ),
        CrosswalkRow(
            framework=fr,
            framework_section="Principle 1.5 - Accountability",
            requirement=(
                "Organizations + individuals developing or "
                "deploying AI should be accountable."
            ),
            trustedrisk_evidence=(
                "Append-only HIPAA audit + RFC 6962 Merkle chain + "
                "data-lineage hash chain + reproducibility SQLite "
                "archive."
            ),
            artefact_path=(
                "src/a2a_agent/{audit,merkle_audit,data_lineage}.py"
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def build_framework_crosswalks() -> FrameworkCrosswalkReport:
    nist = _nist_rows()
    oecd = _oecd_rows()
    return FrameworkCrosswalkReport(
        generated_at=datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        n_rows=len(nist) + len(oecd),
        nist_ai_rmf_rows=nist,
        oecd_ai_principles_rows=oecd,
    )


def render_crosswalks_md(report: FrameworkCrosswalkReport) -> str:
    lines: list[str] = []
    lines.append("# TrustedRisk - NIST AI RMF + OECD AI Principles "
                 "crosswalks")
    lines.append("")
    lines.append(
        f"**Generated**: {report.generated_at} - "
        f"**Total rows**: {report.n_rows} "
        f"(NIST: {len(report.nist_ai_rmf_rows)}, "
        f"OECD: {len(report.oecd_ai_principles_rows)})"
    )
    lines.append("")
    for title, rows in [
        ("NIST AI RMF 1.0 (Jan 2023)", report.nist_ai_rmf_rows),
        ("OECD AI Principles (May 2019)",
         report.oecd_ai_principles_rows),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Section | Requirement | TrustedRisk evidence | "
            "Artefact |"
        )
        lines.append("| --- | --- | --- | --- |")
        for r in rows:
            artefact = (
                f"`{r.artefact_path}`"
                if r.artefact_path else "-"
            )
            lines.append(
                f"| {r.framework_section} | {r.requirement} | "
                f"{r.trustedrisk_evidence} | {artefact} |"
            )
        lines.append("")
    return "\n".join(lines)
