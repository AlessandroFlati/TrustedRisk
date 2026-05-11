"""Composer workflow engine.

A workflow is an ordered list of `WorkflowStep`s. Each step:
  - Identifies the specialist that owns the tool (metadata)
  - Names the async tool function to invoke
  - Declares input bindings, which are either literal values or
    JSON-pointer-style references (`${input.x}`,
    `${steps.<id>.output.<field>}`).

Execution:
  - Inputs are resolved against the live execution context (input dict
    + already-completed step outputs)
  - The tool is called asynchronously
  - Output is captured (via Pydantic .model_dump() when applicable)
  - Stored in the context for downstream resolution

The orchestrator is intentionally minimal — no retries, no parallelism
(steps run sequentially by design — clinical workflows are typically
data-dependent). Phase 6.3 will add an A2A backend that swaps the
direct callable with an HTTP request to the specialist endpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkflowStep:
    """One step in a composer workflow.

    A step is either:
      - A *tool hop* (default): identified by `(specialist, tool_name)`,
        executed via `backend.call(...)`. Returns the tool's Pydantic
        artefact; chain refs read its `model_dump()`.
      - A *macro hop*: identified by `sub_workflow_id`, executes
        another workflow recursively and returns a dict of
        `{inner_step_id: output_dump}` so downstream chain refs can
        reference any sub-step output via
        `${steps.<hop>.output.<inner>.<field>}`.

    The macro hop is what lets the Care Engine compose 17+ specialist
    sub-agents into 50+ end-to-end use cases without rewriting every
    base workflow as a deep chain.
    """

    id: str
    specialist: str
    tool_name: str
    callable: Callable[..., Awaitable[Any]] | None = None
    sub_workflow_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    optional: bool = False


@dataclass(frozen=True)
class Workflow:
    """A named, ordered sequence of WorkflowSteps."""

    id: str
    title: str
    description: str
    steps: list[WorkflowStep]
    required_inputs: list[str] = field(default_factory=list)


@dataclass
class WorkflowStepResult:
    """Captured output of a single step."""

    step_id: str
    specialist: str
    tool_name: str
    duration_ms: float
    output: Any
    output_dump: Any = None
    error: str | None = None
    skipped: bool = False


@dataclass
class WorkflowExecution:
    """Aggregate trace of a workflow run."""

    workflow_id: str
    started_at_ms: float
    completed_at_ms: float
    duration_ms: float
    inputs: dict[str, Any]
    steps: list[WorkflowStepResult]
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Reference resolution
# ─────────────────────────────────────────────────────────────────────


def _navigate(root: Any, path: list[str]) -> Any:
    """Descend a dotted path through nested dicts / Pydantic models /
    objects with attribute access. Returns the value or raises KeyError."""
    cur: Any = root
    for part in path:
        if cur is None:
            raise KeyError(f"path {'.'.join(path)!r} hits None at {part!r}")
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"missing key {part!r} in dict")
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            # Try Pydantic .model_dump() if present
            if hasattr(cur, "model_dump"):
                d = cur.model_dump()
                if part in d:
                    cur = d[part]
                    continue
            raise KeyError(
                f"path {'.'.join(path)!r}: cannot resolve {part!r} on "
                f"{type(cur).__name__}"
            )
    return cur


def _resolve_value(
    expr: Any,
    workflow_input: dict[str, Any],
    step_outputs: dict[str, Any],
) -> Any:
    """Resolve a single input binding.

    String values that match `${...}` are interpreted as references;
    dict and list containers are walked recursively so references
    nested inside literal payloads (e.g. `decision_card.patient_reference`
    in the discharge_planning handoff step) resolve correctly.
    Everything else passes through verbatim.
    """
    if isinstance(expr, dict):
        return {
            k: _resolve_value(v, workflow_input, step_outputs)
            for k, v in expr.items()
        }
    if isinstance(expr, list):
        return [
            _resolve_value(v, workflow_input, step_outputs) for v in expr
        ]
    if not isinstance(expr, str):
        return expr
    if not (expr.startswith("${") and expr.endswith("}")):
        return expr

    inner = expr[2:-1].strip()
    if not inner:
        raise ValueError(f"empty reference: {expr!r}")

    parts = inner.split(".")
    head = parts[0]
    if head == "input":
        return _navigate(workflow_input, parts[1:])
    if head == "steps":
        # ${steps.<step_id>.output[.<field>...]}
        if len(parts) < 3 or parts[2] != "output":
            raise ValueError(
                f"step reference must be of the form "
                f"${{steps.<id>.output[.<field>...]}}; got {expr!r}"
            )
        step_id = parts[1]
        if step_id not in step_outputs:
            raise KeyError(
                f"reference {expr!r}: step {step_id!r} has no output yet"
            )
        return _navigate(step_outputs[step_id], parts[3:])
    raise ValueError(
        f"unknown reference root {head!r} in {expr!r} (expected "
        f"'input' or 'steps')"
    )


def _resolve_inputs(
    raw_inputs: dict[str, Any],
    workflow_input: dict[str, Any],
    step_outputs: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in raw_inputs.items():
        out[k] = _resolve_value(v, workflow_input, step_outputs)
    return out


# ─────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────


def _detect_abstain_in_output(
    output: Any,
) -> tuple[bool, str | None, str | None]:
    """Inspect a step's output for an abstain flag.

    Recognised shapes:
      - Pydantic model with `.abstain_recommended` (single-tool step).
      - Dict with top-level `abstain_recommended` (single-tool step
        where the backend already model_dumped).
      - Macro-hop dict mapping {inner_step_id: inner_dump}: the flag
        lives one level deeper, so a sub-workflow that abstained on
        any inner step still surfaces here.

    Returns `(abstained, origin, reason)`:
      - origin is "(direct)" for shapes 1 and 2, or the inner step_id
        for shape 3.
      - reason is the abstain_reason if available.
    """
    if hasattr(output, "abstain_recommended") and not isinstance(output, dict):
        if getattr(output, "abstain_recommended", False):
            return True, "(direct)", getattr(output, "abstain_reason", None)
        return False, None, None
    if isinstance(output, dict):
        if output.get("abstain_recommended"):
            return True, "(direct)", output.get("abstain_reason")
        for inner_id, inner in output.items():
            if isinstance(inner, dict) and inner.get("abstain_recommended"):
                return True, str(inner_id), inner.get("abstain_reason")
    return False, None, None


def _to_dump(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            return None
    if isinstance(o, dict):
        # Macro hop already returns a navigable dict.
        return o
    return None


async def _execute_macro_hop(
    sub_workflow_id: str,
    resolved_inputs: dict[str, Any],
    backend: Any,
) -> dict[str, Any]:
    """Run a sub-workflow recursively and flatten its trace.

    Returns a `{inner_step_id: output_dump_or_dict}` mapping. The
    caller (the outer execute_workflow loop) stores this as the macro
    hop's output, so subsequent chain references like
    `${steps.hop_2.output.cssrs.risk_level}` resolve naturally through
    the sub-workflow's nested step outputs.
    """
    from .workflows import REGISTRY   # lazy to avoid circular import
    sub_wf = REGISTRY.get(sub_workflow_id)
    if sub_wf is None:
        raise ValueError(
            f"macro hop references unknown sub-workflow id "
            f"{sub_workflow_id!r}; known ids: "
            f"{sorted(REGISTRY.keys())[:5]}..."
        )
    # Sub-workflow inputs inherit from the macro's resolved inputs.
    # The caller (_run_workflow or _run_specialist_route) has already
    # overlaid the live FHIR context on top of the workflow-level inputs,
    # so no demo defaults are needed here. Any still-missing required
    # input is filled with "" as last resort so the tool can abstain
    # cleanly rather than raising a KeyError.
    sub_inputs: dict[str, Any] = {}
    for k, v in resolved_inputs.items():
        if v is not None and v != "":
            sub_inputs[k] = v
    for k in sub_wf.required_inputs:
        sub_inputs.setdefault(k, "")
    sub_exec = await execute_workflow(sub_wf, sub_inputs, backend=backend)
    return {
        s.step_id: (s.output_dump if s.output_dump is not None else
                     {"_error": s.error} if s.error else
                     {"_skipped": s.skipped})
        for s in sub_exec.steps
    }


async def execute_workflow(
    workflow: Workflow,
    inputs: dict[str, Any],
    *,
    stop_on_abstain: bool = True,
    backend: Any | None = None,
) -> WorkflowExecution:
    """Run a workflow end-to-end.

    Args:
        workflow: the Workflow instance.
        inputs: workflow-level inputs (must include every name in
            `workflow.required_inputs`).
        stop_on_abstain: when True, the execution halts as soon as a
            step returns `abstain_recommended=True` (Pydantic-modelled
            outputs only). The result is still returned with the partial
            trace.
        backend: optional WorkflowBackend (see `apps.composer.backends`).
            When None, picks via TRUSTEDRISK_COMPOSER_BACKEND env. The
            in-process backend imports tool callables directly; the A2A
            backend dispatches each step as an HTTP `tools/call` against
            the specialist's MCP endpoint.

    Returns:
        WorkflowExecution with the per-step trace.
    """
    if backend is None:
        from .backends import default_backend
        backend = default_backend()

    missing = [k for k in workflow.required_inputs if k not in inputs]
    if missing:
        raise ValueError(
            f"workflow {workflow.id!r} is missing required input(s): "
            f"{missing!r}"
        )

    started = time.perf_counter()
    started_iso = started * 1000.0

    step_outputs: dict[str, Any] = {}
    step_results: list[WorkflowStepResult] = []
    abstain_recommended = False
    abstain_reason: str | None = None

    # ---- Pure-macro fast path: parallel sub-workflows ------------------
    # A macro built by `_make_macro` has every step shaped as a macro hop
    # with `inputs={}`, no `callable`, and `sub_workflow_id` set. There
    # is no chain reference between hops by construction (each hop only
    # inherits the macro-level inputs), so the hops are independent and
    # can be run concurrently. PO's chat client times out external tool
    # responses on a tight budget (~30 s observed), and a serial macro
    # of 3 sub-workflows easily exceeds 60 s when each sub-workflow
    # itself fetches a FHIR bundle and runs 5 - 10 tool calls.
    # Parallelising the hops brings the wall-clock down to the slowest
    # single sub-workflow, which keeps PO inside its budget.
    is_pure_macro = (
        len(workflow.steps) > 1
        and all(
            s.sub_workflow_id is not None and not s.inputs
            for s in workflow.steps
        )
    )
    if is_pure_macro:
        import asyncio as _asyncio

        async def _run_one_hop(step: WorkflowStep) -> WorkflowStepResult:
            hop_t0 = time.perf_counter()
            try:
                output = await _execute_macro_hop(
                    step.sub_workflow_id, dict(inputs), backend,
                )
            except Exception as exc:
                return WorkflowStepResult(
                    step_id=step.id, specialist=step.specialist,
                    tool_name=step.tool_name,
                    duration_ms=round((time.perf_counter() - hop_t0) * 1000.0, 2),
                    output=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            dt = (time.perf_counter() - hop_t0) * 1000.0
            dump = _to_dump(output)
            return WorkflowStepResult(
                step_id=step.id, specialist=step.specialist,
                tool_name=step.tool_name, duration_ms=round(dt, 2),
                output=output, output_dump=dump,
            )

        step_results = list(await _asyncio.gather(
            *[_run_one_hop(s) for s in workflow.steps],
            return_exceptions=False,
        ))
        # Macro hops are constructed with optional=True, so an inner
        # abstain does not flip the top-level flag (matches the serial
        # path's optional-step contract). Stash dumps into step_outputs
        # for downstream consumers that introspect the WorkflowExecution.
        for sr in step_results:
            if sr.output_dump is not None:
                step_outputs[sr.step_id] = sr.output_dump
            elif sr.output is not None:
                step_outputs[sr.step_id] = sr.output

        completed = time.perf_counter()
        return WorkflowExecution(
            workflow_id=workflow.id,
            started_at_ms=round(started_iso, 2),
            completed_at_ms=round(completed * 1000.0, 2),
            duration_ms=round((completed - started) * 1000.0, 2),
            inputs=inputs,
            steps=step_results,
            abstain_recommended=False,
            abstain_reason=None,
        )

    # ---- Serial path (single workflow with chained step inputs) --------
    for step in workflow.steps:
        try:
            resolved = _resolve_inputs(step.inputs, inputs, step_outputs)
        except (KeyError, ValueError) as exc:
            if step.optional:
                step_results.append(WorkflowStepResult(
                    step_id=step.id, specialist=step.specialist,
                    tool_name=step.tool_name, duration_ms=0.0,
                    output=None, skipped=True,
                    error=f"input resolution failed: {exc}",
                ))
                continue
            raise

        t0 = time.perf_counter()
        try:
            if step.sub_workflow_id is not None:
                # Macro hop: invoke another workflow recursively. The
                # macro's step has inputs={} by construction (the chain
                # is implicit), so `resolved` is empty -- forward the
                # macro's *workflow-level* inputs to the sub-workflow
                # instead, with any per-step resolved override applied
                # on top. Returns a flat {inner_step_id: output_dump}
                # dict so chain refs of the form
                # ${steps.<hop>.output.<inner>.<field>} keep navigating.
                hop_inputs = {**inputs, **resolved}
                output = await _execute_macro_hop(
                    step.sub_workflow_id, hop_inputs, backend,
                )
            else:
                output = await backend.call(
                    step.specialist, step.tool_name, step.callable,
                    resolved,
                )
        except Exception as exc:
            step_results.append(WorkflowStepResult(
                step_id=step.id, specialist=step.specialist,
                tool_name=step.tool_name,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            ))
            if not step.optional:
                # Halt on hard failure of a required step
                break
            continue
        dt = (time.perf_counter() - t0) * 1000.0
        dump = _to_dump(output)
        step_outputs[step.id] = dump if dump is not None else output
        step_results.append(WorkflowStepResult(
            step_id=step.id, specialist=step.specialist,
            tool_name=step.tool_name, duration_ms=round(dt, 2),
            output=output, output_dump=dump,
        ))

        # Soft abstain propagation. Three output shapes can carry an
        # abstain flag:
        #  1) Pydantic model (single-tool step): attribute access.
        #  2) Plain dict with top-level abstain_recommended (single-tool
        #     step where backend already model_dumped).
        #  3) Macro-hop dict of {inner_step_id: inner_output_dump}: the
        #     flag lives one level deeper, on each inner step's dump.
        #
        # An optional step that abstains is *not* a workflow-level
        # abstain: the caller marked it optional precisely because the
        # workflow should keep producing the rest of the bundle even
        # when this signal is unavailable. The abstain remains visible
        # on the per-step output_dump (so the dispatcher's
        # `abstained_steps` list still surfaces it for transparency),
        # but it does NOT flip the top-level `abstain_recommended` flag.
        # Required steps still propagate as before.
        out_abstain, abstain_origin, sub_reason = (
            _detect_abstain_in_output(output)
        )
        if out_abstain and not step.optional:
            abstain_recommended = True
            abstain_reason = (
                f"step {step.id!r} on {step.specialist!r} abstained: "
                f"{abstain_origin}{(': ' + sub_reason) if sub_reason else ''}"
            )
            if stop_on_abstain:
                break

    completed = time.perf_counter()
    return WorkflowExecution(
        workflow_id=workflow.id,
        started_at_ms=round(started_iso, 2),
        completed_at_ms=round(completed * 1000.0, 2),
        duration_ms=round((completed - started) * 1000.0, 2),
        inputs=inputs,
        steps=step_results,
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
    )
