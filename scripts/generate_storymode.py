"""Phase 17.R - Story-mode showcase patients.

6 narrative patient cases that exercise the full TrustedRisk pipeline
(LACE risk -> 3-agent debate -> patient-advocate -> regulatory pack
snippet) and render to a single self-contained HTML at
``docs/showcase/STORYMODE.html``.

Each case is a hand-written 1-2 paragraph chart vignette + structured
demographics + a deterministic forward pass.
"""

from __future__ import annotations

import html as html_lib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.multi_agent_debate import DebateInput, run_debate
from a2a_agent.patient_advocate import (
    PatientAdvocateInput, evaluate_patient_advocate,
)


OUT_DIR = ROOT / "docs" / "showcase"


@dataclass
class StoryPatient:
    case_id: str
    title: str
    chart_text: str
    structured: dict[str, Any]
    risk_point_estimate: float
    risk_ci_width: float
    proposed_action: str
    advocate_input: PatientAdvocateInput


def _patients() -> list[StoryPatient]:
    return [
        StoryPatient(
            case_id="STORY-1",
            title=(
                "Eleanor Thompson, 78F, White, Medicare - "
                "Acute-on-chronic CHF with AKI"
            ),
            chart_text=(
                "78-year-old woman with HFrEF (EF 28%), CKD stage 3, "
                "and atrial fibrillation on apixaban admitted from "
                "the ED for acute-on-chronic systolic heart failure. "
                "On day 3, BUN 38, Cr 1.6 (baseline 1.1), BNP 1240. "
                "Diuresed 5 L. By day 7 she is euvolemic, BUN 24, "
                "Cr 1.2, ambulating with PT, daughter is the primary "
                "caregiver and lives 10 minutes away. She speaks "
                "English natively and asks how to recognize fluid "
                "overload at home. LACE = 9 (L3 A0 C4 E2)."
            ),
            structured={
                "age": 78, "sex": "female", "race": "white",
                "insurance": "medicare", "language": "english",
                "lace": 9, "n_chronic_meds": 7,
                "primary_problem": "acute_on_chronic_chf_aki",
            },
            risk_point_estimate=0.158,
            risk_ci_width=0.045,
            proposed_action="discharge_with_homecare",
            advocate_input=PatientAdvocateInput(
                patient_age=78, patient_race="white",
                patient_insurance="medicare",
                patient_language="english",
                n_chronic_medications=7,
                has_home_caregiver_available=True,
                has_transportation=True,
                fairness_audit_present=True,
                recommended_action="discharge_with_homecare",
                risk_point_estimate=0.158,
            ),
        ),
        StoryPatient(
            case_id="STORY-2",
            title=(
                "Marcus Williams, 62M, Black, Medicaid - "
                "Chest pain rule-out"
            ),
            chart_text=(
                "62-year-old African-American man with hypertension, "
                "T2D, and BMI 32 brought by his daughter for chest "
                "discomfort that started while mowing the lawn. ED "
                "troponin 0.02 (negative), ECG with non-specific T-"
                "wave changes, no prior CAD, three risk factors. "
                "HEART score 4 (history 1, ECG 1, age 1, RF 1, "
                "trop 0). The 4-hour delta troponin is also "
                "negative. He is concerned about missing his factory "
                "shift tomorrow morning + cannot afford a stress "
                "test deductible. LACE = 4 (L1 A0 C2 E1)."
            ),
            structured={
                "age": 62, "sex": "male", "race": "black",
                "insurance": "medicaid", "language": "english",
                "lace": 4, "n_chronic_meds": 4,
                "primary_problem": "non_specific_chest_pain",
            },
            risk_point_estimate=0.103,
            risk_ci_width=0.05,
            proposed_action="discharge_home",
            advocate_input=PatientAdvocateInput(
                patient_age=62, patient_race="black",
                patient_insurance="medicaid",
                patient_language="english",
                n_chronic_medications=4,
                has_home_caregiver_available=True,
                has_transportation=True,
                fairness_audit_present=True,
                recommended_action="discharge_home",
                risk_point_estimate=0.103,
            ),
        ),
        StoryPatient(
            case_id="STORY-3",
            title=(
                "Ana Lucia Hernandez, 34F, Hispanic, uninsured - "
                "Preeclampsia at 34 weeks"
            ),
            chart_text=(
                "34-year-old G2P1 Hispanic woman at 34 weeks "
                "gestation with new-onset BP 158/96, headache, "
                "+3 proteinuria, platelets 95k, AST 78. Diagnosis: "
                "preeclampsia with severe features. After magnesium "
                "sulfate + antihypertensive treatment, BP improves "
                "to 138/88; the team plans betamethasone for "
                "fetal-lung maturation and inpatient monitoring "
                "until 37 weeks. She speaks Spanish primarily, has "
                "no insurance, and asks who will pay for the "
                "extended stay + prenatal vitamins. LACE = 7 "
                "(L4 A0 C1 E2)."
            ),
            structured={
                "age": 34, "sex": "female", "race": "hispanic",
                "insurance": "uninsured", "language": "spanish",
                "lace": 7, "n_chronic_meds": 5,
                "primary_problem": "preeclampsia_severe_features",
            },
            risk_point_estimate=0.158,
            risk_ci_width=0.045,
            proposed_action="continued_admission",
            advocate_input=PatientAdvocateInput(
                patient_age=34, patient_race="hispanic",
                patient_insurance="uninsured",
                patient_language="spanish",
                n_chronic_medications=5,
                has_home_caregiver_available=True,
                has_transportation=True,
                fairness_audit_present=True,
                recommended_action="continued_admission",
                risk_point_estimate=0.158,
            ),
        ),
        StoryPatient(
            case_id="STORY-4",
            title=(
                "Vinh Nguyen, 71M, Vietnamese, Medicare - "
                "COPD exacerbation + OSA"
            ),
            chart_text=(
                "71-year-old man, primary language Vietnamese, "
                "with GOLD 3 COPD and severe OSA on CPAP admitted "
                "for an acute COPD exacerbation. He completed a "
                "5-day prednisone burst + azithromycin and is back "
                "to baseline FEV1 65% predicted. STOP-BANG 7 "
                "consistent with severe OSA; Epworth 14. Lives "
                "alone but has a sister 20 minutes away who can "
                "help with appointments. He requests Vietnamese-"
                "language counseling materials. LACE = 8 "
                "(L3 A3 C2 E0)."
            ),
            structured={
                "age": 71, "sex": "male", "race": "asian",
                "insurance": "medicare", "language": "vietnamese",
                "lace": 8, "n_chronic_meds": 8,
                "primary_problem": "copd_exacerbation_osa",
            },
            risk_point_estimate=0.158,
            risk_ci_width=0.05,
            proposed_action="discharge_with_homecare",
            advocate_input=PatientAdvocateInput(
                patient_age=71, patient_race="asian",
                patient_insurance="medicare",
                patient_language="vietnamese",
                n_chronic_medications=8,
                has_home_caregiver_available=True,
                has_transportation=True,
                fairness_audit_present=True,
                recommended_action="discharge_with_homecare",
                risk_point_estimate=0.158,
            ),
        ),
        StoryPatient(
            case_id="STORY-5",
            title=(
                "Chayton Bear, 58M, Indigenous, Medicaid - "
                "Type 2 DM with diabetic retinopathy + neuropathy"
            ),
            chart_text=(
                "58-year-old Indigenous man (Lakota Nation) with "
                "long-standing T2D (A1c 9.4), proliferative diabetic "
                "retinopathy with macular edema, and DN4-positive "
                "peripheral neuropathy. Admitted with cellulitis of "
                "the right foot, treated with IV vancomycin + "
                "ceftriaxone, podiatry consult ruled out "
                "osteomyelitis. He lives on a reservation 90 "
                "minutes from the nearest endocrinologist; "
                "transportation to follow-up is uncertain; family "
                "support is limited. LACE = 11 (L4 A3 C3 E1)."
            ),
            structured={
                "age": 58, "sex": "male", "race": "indigenous",
                "insurance": "medicaid", "language": "english",
                "lace": 11, "n_chronic_meds": 10,
                "primary_problem": "t2d_complications_cellulitis",
            },
            risk_point_estimate=0.234,
            risk_ci_width=0.05,
            proposed_action="discharge_with_homecare",
            advocate_input=PatientAdvocateInput(
                patient_age=58, patient_race="indigenous",
                patient_insurance="medicaid",
                patient_language="english",
                n_chronic_medications=10,
                has_home_caregiver_available=False,
                has_transportation=False,
                fairness_audit_present=True,
                recommended_action="discharge_with_homecare",
                risk_point_estimate=0.234,
            ),
        ),
        StoryPatient(
            case_id="STORY-6",
            title=(
                "Dr. Sarah Cohen, 45F, White, commercial, no "
                "caregiver - Breast cancer post-AC chemo"
            ),
            chart_text=(
                "45-year-old emergency-medicine attending with "
                "stage IIB ER+/HER2- breast cancer after cycle 3 of "
                "dose-dense AC, admitted for febrile neutropenia "
                "with ANC 200 and a temperature of 38.6 C. After "
                "broad-spectrum antibiotics + filgrastim her ANC "
                "recovers to 1.4k. She lives alone (recently "
                "divorced), no nearby family, plans to drive "
                "herself home, and explicitly declines a home-care "
                "referral citing cost. LACE = 6 (L2 A3 C1 E0)."
            ),
            structured={
                "age": 45, "sex": "female", "race": "white",
                "insurance": "commercial", "language": "english",
                "lace": 6, "n_chronic_meds": 3,
                "primary_problem": "febrile_neutropenia_post_chemo",
            },
            risk_point_estimate=0.103,
            risk_ci_width=0.05,
            proposed_action="discharge_home",
            advocate_input=PatientAdvocateInput(
                patient_age=45, patient_race="white",
                patient_insurance="commercial",
                patient_language="english",
                n_chronic_medications=3,
                has_home_caregiver_available=False,
                has_transportation=True,
                fairness_audit_present=True,
                recommended_action="discharge_home",
                risk_point_estimate=0.103,
            ),
        ),
    ]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustedRisk - Story-mode showcase</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            Helvetica, Arial, sans-serif; margin: 24px;
            background:#0d1117; color:#c9d1d9; max-width: 1100px; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  h2 {{ border-bottom: 1px solid #30363d; padding-bottom: 4px;
         margin-top: 32px; }}
  .chart {{ background:#161b22; border-left: 3px solid #58a6ff;
              padding: 10px 14px; font-style: italic; }}
  .verdict {{ font-weight: 700; padding: 2px 8px; border-radius: 4px; }}
  .v-approved {{ background: #238636; color: #fff; }}
  .v-revise {{ background: #d29922; color: #000; }}
  .v-force-abstain {{ background: #f85149; color: #fff; }}
  .v-concur {{ background: #238636; color: #fff; }}
  .v-challenge {{ background: #d29922; color: #000; }}
  .v-escalate {{ background: #f85149; color: #fff; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #30363d; padding: 4px 8px;
            text-align: left; vertical-align: top; }}
  th {{ background: #161b22; }}
  .pill {{ display:inline-block; background:#1f6feb; color:#fff;
            padding:1px 8px; border-radius:12px; font-size:11px;
            margin-right:4px; }}
  details {{ margin: 6px 0; padding: 6px 12px;
              background:#161b22; border:1px solid #30363d;
              border-radius:4px; }}
  summary {{ cursor: pointer; color:#79c0ff; font-weight:600; }}
</style></head>
<body>
<h1>TrustedRisk - Story-mode showcase</h1>
<p>Generated {timestamp} - {n_cases} narrative patient cases. Each case
runs the full TrustedRisk pipeline: calibrated 30-day readmission risk -&gt;
3-agent debate (clinical_conservative + evidence_aggressive +
fairness_guard) -&gt; patient-advocate (5 axes: fairness, autonomy,
accessibility, financial_burden, language). Subgroup demographics span
race, insurance, language, and accessibility surface to demonstrate
the safety-floor + advocate vetoes in action.</p>
{cases_html}
</body></html>
"""


def _verdict_class(v: str) -> str:
    return f"v-{v.replace('_', '-')}"


def _render_case(p: StoryPatient) -> str:
    payload = DebateInput(
        recommended_action=p.proposed_action,
        risk_point_estimate=p.risk_point_estimate,
        risk_ci_width=p.risk_ci_width,
        fairness_subgroup=(
            (p.advocate_input.patient_race or "").lower()
            if (p.advocate_input.patient_race or "").lower() in (
                "black", "indigenous"
            )
            else (p.advocate_input.patient_insurance or "").lower()
            if (p.advocate_input.patient_insurance or "").lower() in (
                "medicaid", "uninsured"
            )
            else None
        ),
        fairness_audit_present=True,
    )
    debate = run_debate(payload)
    advocate = evaluate_patient_advocate(p.advocate_input)
    pills = " ".join(
        f"<span class=pill>{html_lib.escape(k)}={html_lib.escape(str(v))}"
        f"</span>"
        for k, v in p.structured.items()
    )
    debate_rows = "".join(
        f"<tr><td>{html_lib.escape(v.agent_id)}</td>"
        f"<td><span class='verdict v-{v.vote.replace('_','-')}'>"
        f"{v.vote}</span></td>"
        f"<td>{html_lib.escape(v.rationale)}</td></tr>"
        for v in debate.votes
    )
    advocate_rows = "".join(
        f"<tr><td>{html_lib.escape(f.axis)}</td>"
        f"<td><span class='verdict v-{f.verdict.replace('_','-')}'>"
        f"{f.verdict}</span></td>"
        f"<td>{html_lib.escape(f.rationale)}</td></tr>"
        for f in advocate.findings
    )
    return f"""
<h2>{html_lib.escape(p.case_id)}: {html_lib.escape(p.title)}</h2>
<p>{pills}</p>
<div class="chart">{html_lib.escape(p.chart_text)}</div>
<p><b>Initial proposed action</b>: <code>{html_lib.escape(p.proposed_action)}</code> at
risk {p.risk_point_estimate*100:.1f}% (CI width {p.risk_ci_width*100:.1f}%).</p>
<h3>Multi-agent debate verdict:
<span class="verdict {_verdict_class(debate.verdict)}">{debate.verdict}</span></h3>
<table><thead><tr><th>Agent</th><th>Vote</th><th>Rationale</th></tr></thead>
<tbody>{debate_rows}</tbody></table>
<h3>Patient-advocate verdict:
<span class="verdict {_verdict_class(advocate.overall_verdict)}">
{advocate.overall_verdict}</span></h3>
<table><thead><tr><th>Axis</th><th>Verdict</th><th>Rationale</th></tr></thead>
<tbody>{advocate_rows}</tbody></table>
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cases = _patients()
    cases_html = "\n".join(_render_case(p) for p in cases)
    html = _HTML_TEMPLATE.format(
        timestamp=timestamp, n_cases=len(cases),
        cases_html=cases_html,
    )
    out = OUT_DIR / "STORYMODE.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"{len(cases)} narrative cases rendered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
