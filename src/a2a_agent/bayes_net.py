"""Phase 17.AL - Bayesian network inference for patient state.

Pure-Python implementation of a small discrete Bayesian network +
variable-elimination inference. Designed for clinical-state DAGs
like:

    LACE_band -> Comorbidity_severity
    LACE_band -> Outcome
    Comorbidity_severity -> Outcome
    SocialDeterminants -> Outcome

Each node carries a conditional probability table (CPT). Inference
returns ``P(query | evidence)`` with no NumPy dependency.

References:
- Pearl 1988 - probabilistic reasoning in intelligent systems.
- Koller & Friedman 2009 - probabilistic graphical models.
"""

from __future__ import annotations

import itertools
from typing import Iterable

from pydantic import BaseModel, Field


class BNNode(BaseModel):
    name: str
    states: list[str]
    parents: list[str] = Field(default_factory=list)
    # cpt[ tuple(parent_states) ][ child_state ] = probability
    cpt: dict[str, dict[str, float]] = Field(default_factory=dict)


class BayesianNetwork(BaseModel):
    nodes: dict[str, BNNode]


# ─────────────────────────────────────────────────────────────────────
# CPT key encoding (tuples are not JSON-friendly; we use "|" joins)
# ─────────────────────────────────────────────────────────────────────


def _key(parent_values: tuple[str, ...]) -> str:
    return "|".join(parent_values) if parent_values else ""


def cpt_lookup(node: BNNode, parent_values: tuple[str, ...],
               child_value: str) -> float:
    row = node.cpt.get(_key(parent_values), {})
    return row.get(child_value, 0.0)


# ─────────────────────────────────────────────────────────────────────
# Network builder + validator
# ─────────────────────────────────────────────────────────────────────


def build_network(nodes: list[BNNode]) -> BayesianNetwork:
    by_name = {n.name: n for n in nodes}
    if len(by_name) != len(nodes):
        raise ValueError("duplicate node names")
    # Validate each CPT row sums to ~1
    for n in nodes:
        for missing in n.parents:
            if missing not in by_name:
                raise ValueError(
                    f"node {n.name!r} declares unknown parent "
                    f"{missing!r}"
                )
        # CPT key arity
        if not n.parents:
            row = n.cpt.get("", {})
            total = sum(row.get(s, 0.0) for s in n.states)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"root node {n.name!r} CPT sums to {total}")
        else:
            parent_states = [by_name[p].states for p in n.parents]
            for combo in itertools.product(*parent_states):
                row = n.cpt.get(_key(combo), {})
                total = sum(row.get(s, 0.0) for s in n.states)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"node {n.name!r} CPT row "
                        f"{_key(combo)!r} sums to {total}")
    # Topological order check (no cycles)
    order = _topological_order(by_name)
    if len(order) != len(by_name):
        raise ValueError("network has a cycle")
    return BayesianNetwork(nodes=by_name)


def _topological_order(
    by_name: dict[str, BNNode],
) -> list[str]:
    """Kahn's algorithm; returns a partial order shorter than the
    network when a cycle exists (caller must check the length)."""
    indegree: dict[str, int] = {n: 0 for n in by_name}
    for n, node in by_name.items():
        for p in node.parents:
            if p in indegree:
                indegree[n] += 1
    queue = [n for n, d in indegree.items() if d == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for child, child_node in by_name.items():
            if n in child_node.parents:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
    return order


# ─────────────────────────────────────────────────────────────────────
# Inference - enumeration over the joint
# ─────────────────────────────────────────────────────────────────────


class InferenceResult(BaseModel):
    query: str
    distribution: dict[str, float]
    evidence: dict[str, str]
    rationale: str


def _joint_probability(
    network: BayesianNetwork, assignment: dict[str, str],
) -> float:
    p = 1.0
    for name, node in network.nodes.items():
        if name not in assignment:
            return 0.0
        parent_values = tuple(
            assignment[parent] for parent in node.parents
        )
        p *= cpt_lookup(node, parent_values, assignment[name])
    return p


def infer(
    *,
    network: BayesianNetwork,
    query: str,
    evidence: dict[str, str] | None = None,
) -> InferenceResult:
    """Return P(query | evidence) by enumerating over hidden
    variables. Variables: query name + evidence dict (variable ->
    state).

    Pure-enumeration; complexity O(prod_i |Val(X_i)|) where the
    product is over hidden variables. Fine for the small clinical
    DAGs we use.
    """
    if query not in network.nodes:
        raise ValueError(f"unknown query node: {query!r}")
    evidence = dict(evidence or {})
    for name, value in evidence.items():
        if name not in network.nodes:
            raise ValueError(f"unknown evidence node: {name!r}")
        if value not in network.nodes[name].states:
            raise ValueError(
                f"unknown state {value!r} for node {name!r}")

    hidden = [
        name for name in network.nodes
        if name != query and name not in evidence
    ]
    distribution: dict[str, float] = {}
    for q_state in network.nodes[query].states:
        prob = 0.0
        if hidden:
            domains = [network.nodes[h].states for h in hidden]
            for combo in itertools.product(*domains):
                assignment = {
                    **evidence,
                    query: q_state,
                    **dict(zip(hidden, combo)),
                }
                prob += _joint_probability(network, assignment)
        else:
            assignment = {**evidence, query: q_state}
            prob = _joint_probability(network, assignment)
        distribution[q_state] = prob

    total = sum(distribution.values())
    if total > 0:
        distribution = {
            k: round(v / total, 6) for k, v in distribution.items()
        }
    else:
        distribution = {
            k: 0.0 for k in network.nodes[query].states
        }
    return InferenceResult(
        query=query, distribution=distribution,
        evidence=evidence,
        rationale=(
            f"Variable elimination on a {len(network.nodes)}-node "
            f"network; enumerated {len(hidden)} hidden variable(s); "
            f"normalising mass {total:.6f}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Pre-built clinical readmission DAG
# ─────────────────────────────────────────────────────────────────────


def build_readmission_network() -> BayesianNetwork:
    """A small DAG that mirrors the W1 LACE risk model + adds a
    socioeconomic-determinants node."""
    lace = BNNode(
        name="LACE_band",
        states=["low", "mod", "high", "severe"],
        cpt={"": {
            "low": 0.40, "mod": 0.35,
            "high": 0.18, "severe": 0.07,
        }},
    )
    sdoh = BNNode(
        name="SDoH",
        states=["adequate", "inadequate"],
        cpt={"": {"adequate": 0.7, "inadequate": 0.3}},
    )
    comorbid = BNNode(
        name="Comorbidity",
        states=["mild", "severe"],
        parents=["LACE_band"],
        cpt={
            "low":    {"mild": 0.85, "severe": 0.15},
            "mod":    {"mild": 0.60, "severe": 0.40},
            "high":   {"mild": 0.30, "severe": 0.70},
            "severe": {"mild": 0.10, "severe": 0.90},
        },
    )
    outcome = BNNode(
        name="Outcome",
        states=["no_readmit", "readmit"],
        parents=["LACE_band", "Comorbidity", "SDoH"],
        cpt={
            "low|mild|adequate":      {"no_readmit": 0.95, "readmit": 0.05},
            "low|mild|inadequate":    {"no_readmit": 0.90, "readmit": 0.10},
            "low|severe|adequate":    {"no_readmit": 0.88, "readmit": 0.12},
            "low|severe|inadequate":  {"no_readmit": 0.80, "readmit": 0.20},
            "mod|mild|adequate":      {"no_readmit": 0.90, "readmit": 0.10},
            "mod|mild|inadequate":    {"no_readmit": 0.82, "readmit": 0.18},
            "mod|severe|adequate":    {"no_readmit": 0.78, "readmit": 0.22},
            "mod|severe|inadequate":  {"no_readmit": 0.65, "readmit": 0.35},
            "high|mild|adequate":     {"no_readmit": 0.82, "readmit": 0.18},
            "high|mild|inadequate":   {"no_readmit": 0.70, "readmit": 0.30},
            "high|severe|adequate":   {"no_readmit": 0.65, "readmit": 0.35},
            "high|severe|inadequate": {"no_readmit": 0.50, "readmit": 0.50},
            "severe|mild|adequate":   {"no_readmit": 0.70, "readmit": 0.30},
            "severe|mild|inadequate": {"no_readmit": 0.55, "readmit": 0.45},
            "severe|severe|adequate":   {"no_readmit": 0.50, "readmit": 0.50},
            "severe|severe|inadequate": {"no_readmit": 0.30, "readmit": 0.70},
        },
    )
    return build_network([lace, sdoh, comorbid, outcome])
