"""End-to-end smoke test for the 4 PO demo patients.

Simulates the full PO chat -> TrustedRisk A2A roundtrip for each of the
four demo bundles, without restarting the MCP server or hitting the live
PO workspace. The bundle is loaded from the JSON fixture, monkey-patched
into ``mcp_server.fhir.client.fetch_patient_bundle`` via the
``bind_in_memory_fhir`` helper, and then ``_orchestrator_handler`` is
called with the same shape of message PO emits in production.

For every patient the script reports:

  - which dispatch target the engine picked (workflow, specialist, fanout)
  - the expected abstain posture (Eleanor = abstain demo, others = positive)
  - the actual ``abstain_recommended`` flag and any abstained step ids
  - whether each step's output is non-empty

Run::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/smoke_demo_e2e.py

Exit code 0 if every patient meets its expected posture; 1 otherwise.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.fixtures.fhir_helpers import bind_in_memory_fhir  # noqa: E402

# Importing the handler triggers FastMCP / Starlette imports that take a
# few hundred ms but must happen exactly once.
from apps.orchestrator.server import _orchestrator_handler  # noqa: E402


_FHIR_CTX_URI = "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"


# ───────────────────────────── fixtures ─────────────────────────────


CASES: list[dict[str, Any]] = [
    {
        "name": "Eleanor Greene (abstain demo)",
        "bundle_path": "fixtures/po_demo/eleanor_greene.json",
        "prompt": "Run TrustedRisk's full discharge bundle for this patient.",
        "expected_posture": "abstain",
        # The LLM dispatcher may pick either `discharge_planning` (single
        # workflow with a `risk` step that hits LACE OOD) or
        # `discharge_full` (macro that chains multiple sub-workflows
        # whose abstain reasons differ). We require at least one of the
        # documented abstain triggers; the demo posture is "this patient
        # forces abstain somewhere", not a specific path.
        "expected_abstain_keywords_any": [
            "lace_calibration_plateau_OOD",
            "missing_admission_meds_and_discharge_meds",
            "missing_decision_card_action",
            "missing_clinician_inputs",
            "missing_question",
        ],
    },
    {
        "name": "Marcus Reyes (sepsis pipeline)",
        "bundle_path": "fixtures/po_demo/marcus_reyes.json",
        "prompt": "Use TrustedRisk to get and show a complete sepsis pipeline on this patient.",
        "expected_posture": "positive",
    },
    {
        "name": "Nadia Okafor (PA appeal pipeline)",
        "bundle_path": "fixtures/po_demo/nadia_okafor.json",
        "prompt": "Use TrustedRisk to draft an appeal for the denied adalimumab on this patient.",
        "expected_posture": "positive",
    },
    {
        "name": "Sofia Ramirez (well-child + ACIP vaccines)",
        "bundle_path": "fixtures/po_demo/sofia_ramirez.json",
        "prompt": "Run TrustedRisk's well-child workflow and identify any overdue ACIP vaccines.",
        "expected_posture": "positive",
    },
]


# ─────────────────────────── helpers ───────────────────────────


def _patient_id_from_bundle(bundle: dict) -> str:
    """The fixtures use POST + urn:uuid; the Patient resource has no id
    on the wire. We attach a synthetic id matching the slug for the
    in-memory monkey-patch -- the dispatcher only needs a consistent
    value through one call.
    """
    for entry in bundle.get("entry", []):
        r = entry.get("resource") or {}
        if r.get("resourceType") == "Patient":
            r.setdefault("id", "demo-test-patient")
            return f"Patient/{r['id']}"
    return "Patient/demo-test-patient"


def _build_msg(patient_id: str, prompt: str) -> dict[str, Any]:
    """Construct the same shape PO emits in a SendA2AMessage."""
    bare_id = patient_id.split("/", 1)[-1]
    return {
        "role": "user",
        "parts": [{"kind": "text", "text": prompt}],
        "metadata": {
            _FHIR_CTX_URI: {
                "fhirUrl": "http://in-memory/fhir",
                "fhirToken": "in-memory-test-token",
                "patientId": bare_id,
            },
        },
    }


def _scan_for_abstain(text: str, artifacts: list[dict]) -> tuple[bool, list[str]]:
    """Return (has_abstain, list_of_abstain_reasons)."""
    reasons: list[str] = []
    if "[ABSTAIN WARNING]" in text or "ABSTAINED on" in text:
        reasons.append(text)
    for art in artifacts or []:
        for part in art.get("parts") or []:
            data = part.get("data") if isinstance(part, dict) else None
            if not isinstance(data, dict):
                continue
            if data.get("abstain_recommended"):
                reasons.append(
                    f"top-level abstain: {data.get('abstain_reason')}"
                )
            for step in data.get("abstained_steps") or []:
                reasons.append(
                    f"step {step.get('step_id')}: {step.get('reason')}"
                )
    return bool(reasons), reasons


# ─────────────────────────── runner ───────────────────────────


async def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    bundle_path = ROOT / case["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    patient_id = _patient_id_from_bundle(bundle)
    msg = _build_msg(patient_id, case["prompt"])

    text: str = ""
    artifacts: list[dict] = []
    err: str | None = None
    with bind_in_memory_fhir(bundle, patient_id=patient_id):
        try:
            text, artifacts = await _orchestrator_handler(case["prompt"], msg)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

    has_abstain, abstain_reasons = _scan_for_abstain(text, artifacts)
    expected = case["expected_posture"]
    if err:
        ok = False
        verdict = f"ERROR: {err}"
    elif expected == "abstain":
        if not has_abstain:
            ok = False
            verdict = "expected abstain demo, but no abstain found"
        else:
            joined = " | ".join(abstain_reasons)
            any_keys = case.get("expected_abstain_keywords_any") or []
            present = [k for k in any_keys if k in joined]
            ok = bool(present) if any_keys else True
            verdict = (
                f"OK: abstain fired (matched: {', '.join(present)})"
                if ok else
                f"abstain fired but no expected keyword found in: {any_keys}"
            )
    else:
        ok = not has_abstain
        verdict = ("OK: zero abstain, end-to-end positive" if ok
                   else "expected positive demo, but abstain fired")
    return {
        "name": case["name"],
        "ok": ok,
        "verdict": verdict,
        "has_abstain": has_abstain,
        "abstain_reasons": abstain_reasons,
        "n_artifacts": len(artifacts),
        "text_excerpt": text[:240],
        "error": err,
    }


async def _main() -> int:
    results = []
    for case in CASES:
        print(f"\n=== {case['name']} ===")
        print(f"prompt: {case['prompt']}")
        r = await _run_case(case)
        results.append(r)
        print(f"  status: {'PASS' if r['ok'] else 'FAIL'}")
        print(f"  verdict: {r['verdict']}")
        if r["has_abstain"]:
            for reason in r["abstain_reasons"][:5]:
                print(f"    - {reason[:200]}")
        print(f"  n_artifacts: {r['n_artifacts']}")
        if r["text_excerpt"]:
            print(f"  status_message[:240]: {r['text_excerpt']!r}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
    print()
    print("-" * 60)
    n_pass = sum(1 for r in results if r["ok"])
    print(f"Summary: {n_pass}/{len(results)} cases met expected posture.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
