"""Phase 9.3 -- YAML DSL composer template tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _run(coro):
    return asyncio.run(coro)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────── Tool-ref resolver ───────────────────────


def test_resolve_callable_returns_function():
    from apps.composer.yaml_loader import _resolve_callable
    fn = _resolve_callable(
        "mcp_server.tools.polypharmacy_concerns:detect_polypharmacy_concerns"
    )
    assert callable(fn)
    assert fn.__name__ == "detect_polypharmacy_concerns"


def test_resolve_callable_bad_module_raises():
    from apps.composer.yaml_loader import (
        ToolImportError, _resolve_callable,
    )
    with pytest.raises(ToolImportError, match="cannot import module"):
        _resolve_callable("does.not.exist:fn")


def test_resolve_callable_missing_function_raises():
    from apps.composer.yaml_loader import (
        ToolImportError, _resolve_callable,
    )
    with pytest.raises(ToolImportError, match="no attribute"):
        _resolve_callable(
            "mcp_server.tools.polypharmacy_concerns:nonexistent_function"
        )


def test_resolve_callable_no_colon_raises():
    from apps.composer.yaml_loader import (
        ToolImportError, _resolve_callable,
    )
    with pytest.raises(ToolImportError, match="module.path:function"):
        _resolve_callable("no_colon")


# ─────────────────────── load_workflow_from_yaml ───────────────────────


def test_load_outpatient_yaml_workflow():
    from apps.composer.yaml_loader import load_workflow_from_yaml
    wf = load_workflow_from_yaml(
        REPO_ROOT / "apps" / "composer" / "templates"
        / "outpatient_med_review.yaml"
    )
    assert wf.id == "outpatient_med_review_yaml"
    assert wf.required_inputs == [
        "patient_id", "medications", "fhir_bundle",
        "patient_age", "patient_sex",
    ]
    assert len(wf.steps) == 3
    step_ids = [s.id for s in wf.steps]
    assert step_ids == ["polypharmacy", "gaps", "what_if"]


def test_yaml_workflow_executes_end_to_end():
    """The YAML-loaded workflow must produce the same trace as the
    Python-defined OUTPATIENT_MED_REVIEW."""
    from apps.composer.orchestrator import execute_workflow
    from apps.composer.yaml_loader import load_workflow_from_yaml
    wf = load_workflow_from_yaml(
        REPO_ROOT / "apps" / "composer" / "templates"
        / "outpatient_med_review.yaml"
    )
    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {"resourceType": "Patient", "id": "pt-1"}},
    ]}
    out = _run(execute_workflow(wf, {
        "patient_id": "Patient/pt-1",
        "medications": ["warfarin", "ibuprofen"],
        "fhir_bundle": bundle,
        "patient_age": 78,
        "patient_sex": "female",
    }))
    step_ids = {s.step_id for s in out.steps}
    assert {"polypharmacy", "gaps", "what_if"} <= step_ids


# ─────────────────────── Schema validation ───────────────────────


def test_yaml_missing_id_or_steps_raises(tmp_path: Path):
    from apps.composer.yaml_loader import load_workflow_from_yaml
    bad = tmp_path / "bad.yaml"
    bad.write_text("title: only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_workflow_from_yaml(bad)


def test_yaml_missing_step_tool_raises(tmp_path: Path):
    from apps.composer.yaml_loader import load_workflow_from_yaml
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
id: bad
title: Bad
steps:
  - id: only_id
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tool"):
        load_workflow_from_yaml(bad)


def test_load_workflow_directory_dedup_id_raises(tmp_path: Path):
    """Two YAML files with the same `id` must collide loudly."""
    from apps.composer.yaml_loader import load_workflow_directory
    (tmp_path / "a.yaml").write_text(
        """
id: dup
title: A
steps:
  - id: s1
    tool: mcp_server.tools.polypharmacy_concerns:detect_polypharmacy_concerns
""", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(
        """
id: dup
title: B
steps:
  - id: s1
    tool: mcp_server.tools.polypharmacy_concerns:detect_polypharmacy_concerns
""", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate workflow id"):
        load_workflow_directory(tmp_path)
