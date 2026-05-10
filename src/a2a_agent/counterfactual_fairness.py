"""Phase 17.AR - Counterfactual fairness audit (Kusner 2017).

Kusner, Loftus, Russell, Silva 2017 "Counterfactual Fairness"
(NeurIPS). A predictor ``Y_hat(X, A)`` is *counterfactually fair*
toward protected attribute ``A`` if, for every individual ``u``,

    Y_hat(X(u), A=a) = Y_hat(X(u, A<-a'), A=a')

i.e. the prediction is unchanged under a counterfactual world
where ``A`` is flipped (and the *causally downstream* features of
``X`` are adjusted via a structural causal model).

This module implements the audit:

  - ``StructuralCausalModel`` - a small DAG with ``A`` (protected)
    affecting downstream features ``X_descendants``. The user
    supplies *structural equations* that map (A, ancestors, noise)
    to descendant values.
  - ``flip_protected_attribute`` - run the SCM in the abducted-
    perturbed-predicted (Pearl 2009) three-step algorithm to
    produce the counterfactual feature dict.
  - ``audit_counterfactual_fairness`` - sweep a cohort, score
    in the actual world + the counterfactual world, report the
    distribution of |Y_hat actual - Y_hat counterfactual|.

Pure-Python deterministic.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field


_Predictor = Callable[[dict[str, float]], float]
_StructEq = Callable[[str, dict[str, float]], float]
"""(protected_value, ancestors_dict) -> descendant feature value."""


class StructuralCausalModel(BaseModel):
    """A minimal SCM:

      - ``protected_attribute``: the variable name of A.
      - ``protected_values``: the discrete domain of A (e.g.
        ['white', 'black']).
      - ``descendants``: list of variable names whose values are
        causally produced by ``A`` (and possibly upstream).
      - The structural equations are passed to
        ``flip_protected_attribute`` as a callable map.
    """
    protected_attribute: str
    protected_values: list[str]
    descendants: list[str]
    independent_features: list[str] = Field(default_factory=list)


def flip_protected_attribute(
    *,
    instance: dict[str, float],
    scm: StructuralCausalModel,
    target_value: str,
    structural_equations: dict[str, _StructEq],
) -> dict[str, float]:
    """Pearl 2009 abduction-action-prediction: produce the
    counterfactual feature dict where ``A`` is set to
    ``target_value``."""
    if scm.protected_attribute not in instance:
        raise ValueError(
            f"instance is missing the protected attribute "
            f"{scm.protected_attribute!r}"
        )
    if target_value not in scm.protected_values:
        raise ValueError(
            f"target_value {target_value!r} not in domain "
            f"{scm.protected_values}"
        )
    cf = dict(instance)
    cf[scm.protected_attribute] = target_value
    # Re-run structural equations for every descendant in topological
    # order. The supplied dict maps descendant name -> equation.
    for desc in scm.descendants:
        eq = structural_equations.get(desc)
        if eq is None:
            continue
        ancestors = {
            k: v for k, v in cf.items()
            if k != scm.protected_attribute and k != desc
        }
        cf[desc] = float(eq(target_value, ancestors))
    return cf


# ─────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────


class CFFairnessFinding(BaseModel):
    instance_id: int
    actual_protected_value: str
    counterfactual_protected_value: str
    actual_score: float
    counterfactual_score: float
    abs_delta: float = Field(ge=0.0)
    flipped_class: bool


class CFFairnessReport(BaseModel):
    n_instances: int = Field(ge=0)
    protected_attribute: str
    threshold_for_class_flip: float
    pct_strict_cf_fair: float = Field(ge=0.0, le=1.0)
    pct_class_flipped: float = Field(ge=0.0, le=1.0)
    mean_abs_delta: float = Field(ge=0.0)
    p95_abs_delta: float = Field(ge=0.0)
    findings: list[CFFairnessFinding]
    rationale: str


def audit_counterfactual_fairness(
    *,
    instances: list[dict[str, float]],
    predictor: _Predictor,
    scm: StructuralCausalModel,
    structural_equations: dict[str, _StructEq],
    flip_target_lookup: dict[str, str] | None = None,
    decision_threshold: float = 0.20,
    strict_cf_tolerance: float = 0.01,
) -> CFFairnessReport:
    """Run the counterfactual-fairness audit over a cohort.

    For each instance, flip the protected attribute to
    ``flip_target_lookup[actual_value]`` (defaulting to "the other"
    value when the domain is binary), run the SCM to produce the
    counterfactual feature dict, score it, and record the delta.
    """
    if not instances:
        raise ValueError("instances cannot be empty")
    if scm.protected_attribute not in instances[0]:
        raise ValueError(
            "instances missing the SCM protected attribute"
        )
    findings: list[CFFairnessFinding] = []
    for i, inst in enumerate(instances):
        actual = str(inst[scm.protected_attribute])
        if flip_target_lookup is not None:
            target = flip_target_lookup.get(actual)
            if target is None:
                continue
        else:
            other = [
                v for v in scm.protected_values if v != actual
            ]
            if not other:
                continue
            target = other[0]
        cf = flip_protected_attribute(
            instance=inst, scm=scm,
            target_value=target,
            structural_equations=structural_equations,
        )
        actual_score = float(predictor(inst))
        cf_score = float(predictor(cf))
        delta = abs(actual_score - cf_score)
        flipped = (
            (actual_score >= decision_threshold)
            != (cf_score >= decision_threshold)
        )
        findings.append(CFFairnessFinding(
            instance_id=i,
            actual_protected_value=actual,
            counterfactual_protected_value=target,
            actual_score=round(actual_score, 6),
            counterfactual_score=round(cf_score, 6),
            abs_delta=round(delta, 6),
            flipped_class=flipped,
        ))
    n = len(findings)
    n_strict = sum(
        1 for f in findings if f.abs_delta <= strict_cf_tolerance
    )
    n_flipped = sum(1 for f in findings if f.flipped_class)
    deltas = sorted(f.abs_delta for f in findings)
    mean_abs = sum(deltas) / n if n else 0.0
    p95 = (
        deltas[min(n - 1, int(0.95 * (n - 1)))] if n else 0.0
    )
    return CFFairnessReport(
        n_instances=n,
        protected_attribute=scm.protected_attribute,
        threshold_for_class_flip=decision_threshold,
        pct_strict_cf_fair=round(n_strict / n if n else 0.0, 6),
        pct_class_flipped=round(n_flipped / n if n else 0.0, 6),
        mean_abs_delta=round(mean_abs, 6),
        p95_abs_delta=round(p95, 6),
        findings=findings,
        rationale=(
            f"Counterfactual-fairness audit on n={n}; "
            f"strict CF-fair (|delta|<={strict_cf_tolerance}) on "
            f"{n_strict}/{n}; class-flipped on {n_flipped}/{n}."
        ),
    )
