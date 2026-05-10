"""healthcare.compute_fairness_advanced -- Phase 13.2 I1.

MCP wrapper for the Equalized Odds + Demographic Parity audit. Exposes
the deterministic fairness module from `a2a_agent.fairness_advanced`
as an MCP-callable tool with structured input/output.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import FairnessAdvancedReport


async def compute_fairness_advanced(
    sensitive_attribute: str,
    rows: list[dict[str, Any]],
    *,
    subgroup_field: str = "subgroup",
    predicted_field: str = "predicted_positive",
    actual_field: str = "actual_positive",
    n_bootstrap: int = 1000,
    seed: int = 4242,
) -> FairnessAdvancedReport:
    """Audit demographic-parity + equalized-odds across subgroups.

    Args:
        sensitive_attribute: free-text label of the protected attribute
            (e.g. "race", "sex", "insurance_band").
        rows: list of dicts; each row carries `subgroup`, `predicted_positive`,
            `actual_positive`. Override the field names with the kwargs
            below if your row shape differs.
        subgroup_field: dict key for the subgroup label (default "subgroup").
        predicted_field: dict key for the binary model prediction.
        actual_field: dict key for the binary ground-truth outcome.
        n_bootstrap: bootstrap samples for the per-subgroup selection
            rate CI95 (default 1000).
        seed: RNG seed for reproducibility (default 4242).

    Returns:
        FairnessAdvancedReport with per-subgroup TPR/FPR/selection rate +
        DP / EOO gaps + chi² p-value + posture.
    """
    from a2a_agent.fairness_advanced import (
        compute_fairness_advanced as _impl,
    )
    return _impl(
        sensitive_attribute=sensitive_attribute,
        rows=rows,
        subgroup_field=subgroup_field,
        predicted_field=predicted_field,
        actual_field=actual_field,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def register(mcp) -> None:
    mcp.tool()(compute_fairness_advanced)
