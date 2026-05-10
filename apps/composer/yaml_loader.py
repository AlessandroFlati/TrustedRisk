"""Phase 9.3 — YAML DSL for composer workflow templates.

Lets a workspace admin author / version-control workflows as YAML
without touching Python. Each YAML file is parsed into the same
`Workflow` model that the in-process Python templates use.

YAML schema (single workflow per file):

    id: my_workflow
    title: A friendly title
    description: Multi-line description.
    required_inputs:
      - patient_id
      - chief_complaint
    steps:
      - id: triage
        specialist: trustedrisk-acute
        tool: mcp_server.tools.admission_triage:compute_admission_triage
        description: ED triage — ESI level + disposition
        optional: false
        inputs:
          chief_complaint: ${input.chief_complaint}
          patient_id: ${input.patient_id}
      - id: ddx
        specialist: trustedrisk-evidence
        tool: mcp_server.tools.differential_diagnosis_ranker:compute_differential_diagnosis_ranker
        inputs:
          chief_complaint: ${steps.triage.output.chief_complaint}
          ...

The `tool` field is a `<module_path>:<function_name>` string; the
loader does an importlib lookup at parse time. A YAML that references
an unknown tool fails fast with `ToolImportError`.

Variable interpolation (`${input.x}`, `${steps.<id>.output.<field>}`)
is unchanged from the Python orchestrator — same resolver, same
semantics.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from .orchestrator import Workflow, WorkflowStep


class ToolImportError(ValueError):
    """Raised when a YAML step's `tool:` reference cannot be imported."""


def _resolve_callable(tool_ref: str):
    """Resolve `module.path:function_name` to the callable."""
    if ":" not in tool_ref:
        raise ToolImportError(
            f"tool reference {tool_ref!r} must use 'module.path:function' "
            "syntax."
        )
    module_path, func_name = tool_ref.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ToolImportError(
            f"cannot import module {module_path!r}: {exc}"
        ) from exc
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ToolImportError(
            f"module {module_path!r} has no attribute {func_name!r}"
        )
    if not callable(fn):
        raise ToolImportError(
            f"{tool_ref!r} resolved to non-callable {type(fn).__name__}"
        )
    return fn


def load_workflow_from_yaml(yaml_path: str | Path) -> Workflow:
    """Parse a YAML file into a Workflow.

    Args:
        yaml_path: path to a single-workflow YAML file.

    Returns:
        Workflow with the same shape as the in-process Python templates.

    Raises:
        ToolImportError when a step's `tool:` reference is bad.
        ValueError on schema violations.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML workflow file not found: {path!s}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML root must be a mapping; got {type(raw).__name__}"
        )

    if "id" not in raw or "steps" not in raw:
        raise ValueError(
            "YAML workflow must have at minimum `id` and `steps` fields."
        )

    steps: list[WorkflowStep] = []
    for raw_step in raw.get("steps", []):
        if not isinstance(raw_step, dict):
            raise ValueError(
                f"each step must be a mapping; got {type(raw_step).__name__}"
            )
        if "id" not in raw_step or "tool" not in raw_step:
            raise ValueError(
                f"step missing required fields (id, tool): {raw_step!r}"
            )
        callable_fn = _resolve_callable(raw_step["tool"])
        # `tool_name` is a human-readable string; default to the func name
        tool_name = raw_step.get("tool_name") or raw_step["tool"].split(":")[1]
        steps.append(WorkflowStep(
            id=raw_step["id"],
            specialist=raw_step.get("specialist", "trustedrisk-mcp"),
            tool_name=tool_name,
            callable=callable_fn,
            inputs=dict(raw_step.get("inputs", {})),
            description=raw_step.get("description", ""),
            optional=bool(raw_step.get("optional", False)),
        ))

    return Workflow(
        id=raw["id"],
        title=raw.get("title", raw["id"]),
        description=raw.get("description", ""),
        steps=steps,
        required_inputs=list(raw.get("required_inputs", [])),
    )


def load_workflow_directory(directory: str | Path) -> dict[str, Workflow]:
    """Load every `*.yaml` / `*.yml` workflow under a directory.

    Each file produces one `Workflow`; the dict is keyed by workflow id.
    Filenames don't have to match `id` — collisions raise.
    """
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"workflow directory not found: {base!s}")
    out: dict[str, Workflow] = {}
    for path in sorted(base.glob("*.y*ml")):
        wf = load_workflow_from_yaml(path)
        if wf.id in out:
            raise ValueError(
                f"duplicate workflow id {wf.id!r} (second occurrence "
                f"in {path!s})"
            )
        out[wf.id] = wf
    return out
