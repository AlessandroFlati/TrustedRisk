"""TrustedRisk composer — Phase 6.1 BYO-orchestrator (port 8780).

Demonstrates the A2A v1 *Agent Composition* pattern: the composer
runs realistic clinical workflows that consult multiple specialists in
sequence. Each workflow template is a declarative list of steps; each
step names a specialist + tool + input mapping. The orchestrator
resolves `${input.x}` and `${steps.<id>.output.<field>}` references at
execution time, calls the tool, and stores the output for downstream
steps.

In-process mode (default): tool functions are imported directly from
`mcp_server.tools` and invoked. Fast, deterministic, ideal for tests.
A2A mode (Phase 6.3 Cloud Run): tool calls become HTTP requests to the
specialist endpoints over A2A. Same orchestrator, swappable backend.
"""

from .orchestrator import (
    Workflow,
    WorkflowExecution,
    WorkflowStep,
    WorkflowStepResult,
    execute_workflow,
)
from .server import app
from .workflows import REGISTRY

__all__ = [
    "Workflow",
    "WorkflowExecution",
    "WorkflowStep",
    "WorkflowStepResult",
    "execute_workflow",
    "REGISTRY",
    "app",
]
