"""CMP-3 -- Data lineage tracker.

Walks the project's data/ directory, hashes each known artifact, and emits
an explicit DAG showing how raw inputs flow through preprocessing to
calibrated coefficients to tool outputs. The graph carries a top-level
`integrity_hash` so any change in any artifact's bytes is detectable.

The DAG ships in two formats:
  - Structured JSON (DataLineageGraph schema)
  - Mermaid graph text (paste-able into any Markdown renderer)

This is NOT a runtime dependency tracer -- it's a release-time snapshot
useful for:
  - audits (regulators want a reproducible chain from data -> model -> output)
  - drift investigations (the integrity_hash diff localizes the change)
  - onboarding (engineers can see the data graph without reading every script)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from shared.schemas import (
    DataLineageEdge,
    DataLineageGraph,
    DataLineageNode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _sha256_of_file(path: Path, chunk_size: int = 64 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _node_for_path(node_id: str, path: Path, *, node_type: str,
                      description: str) -> DataLineageNode:
    sha = _sha256_of_file(path)
    if path.exists() and path.is_file():
        size = path.stat().st_size
        mtime_dt = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc)
        mtime_iso = mtime_dt.isoformat()
    else:
        size = None
        mtime_iso = None
    rel = path.relative_to(PROJECT_ROOT) if path.exists() else path
    return DataLineageNode(
        node_id=node_id,
        node_type=node_type,                     # type: ignore[arg-type]
        description=description,
        sha256=sha,
        size_bytes=size,
        last_modified_iso=mtime_iso,
        path=str(rel) if path.exists() else str(path),
    )


def _virtual_node(node_id: str, *, node_type: str,
                    description: str) -> DataLineageNode:
    return DataLineageNode(
        node_id=node_id,
        node_type=node_type,                     # type: ignore[arg-type]
        description=description,
    )


def _build_nodes() -> list[DataLineageNode]:
    """Build the canonical lineage node set from project artifacts."""
    data = PROJECT_ROOT / "data"
    nodes: list[DataLineageNode] = []

    nodes.append(_virtual_node(
        "mimic_iv_demo_v2.2",
        node_type="raw_dataset",
        description=(
            "MIMIC-IV demo (open access, v2.2) -- physionet.org/content/"
            "mimic-iv-demo/2.2/. Hospital admissions / labs / diagnoses "
            "for 100 patients."),
    ))
    nodes.append(_node_for_path(
        "mimic_iv_lace_cohort_csv",
        data / "mimic_iv_lace_cohort.csv",
        node_type="derived_artifact",
        description=(
            "Pre-processed MIMIC-IV cohort with LACE features extracted, "
            "split into 5 LACE buckets -- the calibration training set."),
    ))
    nodes.append(_node_for_path(
        "coefficients_json",
        data / "coefficients.json",
        node_type="coefficient_bundle",
        description=(
            "Calibrated Beta-Binomial posterior (5 LACE buckets) -- the "
            "runtime coefficients consumed by compute_readmission_risk."),
    ))
    nodes.append(_node_for_path(
        "abstain_policy_json",
        data / "abstain_policy_lace_percentile.json",
        node_type="coefficient_bundle",
        description=(
            "Abstain policy thresholds (LACE percentile bands) -- drives "
            "the deterministic safety gate."),
    ))
    nodes.append(_node_for_path(
        "fairness_baseline_json",
        data / "fairness_baseline.json",
        node_type="coefficient_bundle",
        description=(
            "Subgroup baseline rates (HCUP / AHRQ literature) consumed by "
            "compute_fairness_audit."),
    ))
    nodes.append(_node_for_path(
        "grounding_index_mpnet",
        data / "grounding_index_mpnet.pkl",
        node_type="external_corpus",
        description=(
            "FAISS index of the guideline corpus for ground_claim "
            "retrieval (sentence-transformers/all-mpnet-base-v2)."),
    ))

    # Code modules
    nodes.append(_node_for_path(
        "module_readmission_risk",
        PROJECT_ROOT / "src" / "mcp_server" / "tools" / "readmission_risk.py",
        node_type="code_module",
        description="MCP tool: compute_readmission_risk."))
    nodes.append(_node_for_path(
        "module_fairness_audit",
        PROJECT_ROOT / "src" / "mcp_server" / "tools" / "fairness_audit.py",
        node_type="code_module",
        description="MCP tool: compute_fairness_audit."))
    nodes.append(_node_for_path(
        "module_ground_claim",
        PROJECT_ROOT / "src" / "mcp_server" / "tools" / "ground_claim.py",
        node_type="code_module",
        description="MCP tool: ground_claim."))
    nodes.append(_node_for_path(
        "module_audit",
        PROJECT_ROOT / "src" / "a2a_agent" / "audit.py",
        node_type="code_module",
        description="HIPAA-style audit log + reproducibility archive."))

    # Outputs
    nodes.append(_virtual_node(
        "tool_output_risk_estimate",
        node_type="tool_output",
        description=(
            "RiskEstimate emitted at runtime -- passed to the A2A agent "
            "and downstream partner agents."),
    ))
    nodes.append(_virtual_node(
        "tool_output_decision_card",
        node_type="tool_output",
        description=(
            "DecisionCard composed by the A2A agent and archived in the "
            "reproducibility store."),
    ))
    nodes.append(_node_for_path(
        "reproducibility_archive",
        data / "reproducibility_archive.sqlite3",
        node_type="test_artifact",
        description=(
            "SQLite archive of (request_id -> inputs/outputs/coefficient "
            "version) for replay verification."),
    ))

    return nodes


def _build_edges() -> list[DataLineageEdge]:
    edges = [
        ("mimic_iv_demo_v2.2", "mimic_iv_lace_cohort_csv",
         "transforms", "preprocessing -> LACE features"),
        ("mimic_iv_lace_cohort_csv", "coefficients_json",
         "calibrates", "Beta-Binomial fit per LACE bucket"),
        ("mimic_iv_lace_cohort_csv", "abstain_policy_json",
         "calibrates", "LACE percentile thresholds"),
        ("coefficients_json", "module_readmission_risk",
         "consumed_by", "loaded at runtime by compute_readmission_risk"),
        ("abstain_policy_json", "module_readmission_risk",
         "consumed_by", "abstain triggers"),
        ("fairness_baseline_json", "module_fairness_audit",
         "consumed_by", "subgroup calibration"),
        ("grounding_index_mpnet", "module_ground_claim",
         "consumed_by", "FAISS retrieval"),
        ("module_readmission_risk", "tool_output_risk_estimate",
         "produces", ""),
        ("tool_output_risk_estimate", "tool_output_decision_card",
         "consumed_by", "A2A agent composition"),
        ("module_audit", "reproducibility_archive",
         "produces", "archive_decision()"),
        ("tool_output_decision_card", "reproducibility_archive",
         "consumed_by", "stored for byte-identical replay"),
    ]
    return [DataLineageEdge(source_id=s, target_id=t,
                                relationship=r,    # type: ignore[arg-type]
                                description=d)
            for (s, t, r, d) in edges]


def _mermaid(nodes: list[DataLineageNode],
              edges: list[DataLineageEdge]) -> str:
    """Render a Mermaid `graph LR` text representation."""
    style_for = {
        "raw_dataset": "fill:#fef3c7,stroke:#92400e",
        "derived_artifact": "fill:#dbeafe,stroke:#1e40af",
        "coefficient_bundle": "fill:#dcfce7,stroke:#166534",
        "code_module": "fill:#f3e8ff,stroke:#6b21a8",
        "tool_output": "fill:#fce7f3,stroke:#9f1239",
        "test_artifact": "fill:#e5e7eb,stroke:#374151",
        "external_corpus": "fill:#ffedd5,stroke:#9a3412",
    }
    lines = ["graph LR"]
    for n in nodes:
        label = f"{n.node_id}<br/><i>{n.node_type}</i>"
        lines.append(f'    {n.node_id}["{label}"]')
    for n in nodes:
        st = style_for.get(n.node_type)
        if st:
            lines.append(f"    style {n.node_id} {st}")
    for e in edges:
        rel_label = e.relationship
        lines.append(f"    {e.source_id} -->|{rel_label}| {e.target_id}")
    return "\n".join(lines)


def _integrity_hash(nodes: Iterable[DataLineageNode]) -> str:
    """Hash chain over (node_id, sha256) sorted by node_id.

    Missing artifacts (sha=None) hash as the literal string "<missing>"
    so the integrity hash still detects a deletion or move.
    """
    h = hashlib.sha256()
    for n in sorted(nodes, key=lambda x: x.node_id):
        sha = n.sha256 or "<missing>"
        h.update(f"{n.node_id}\t{sha}\n".encode("utf-8"))
    return h.hexdigest()


def compute_data_lineage() -> DataLineageGraph:
    """Compute the data lineage DAG from the current filesystem snapshot."""
    nodes = _build_nodes()
    edges = _build_edges()
    mermaid = _mermaid(nodes, edges)
    return DataLineageGraph(
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
        nodes=nodes,
        edges=edges,
        n_nodes=len(nodes),
        n_edges=len(edges),
        integrity_hash=_integrity_hash(nodes),
        mermaid_text=mermaid,
    )
