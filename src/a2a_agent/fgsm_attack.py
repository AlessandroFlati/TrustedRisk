"""Phase 17.AK - FGSM-style adversarial attack.

Implements the Fast Gradient Sign Method (Goodfellow et al. 2014)
adapted to the TrustedRisk LACE classifier:

    x_adv = x + epsilon * sign(grad_x f(x))

The model emitted by the W1 calibration is non-differentiable (a
piecewise-constant bucket map LACE -> P), so we approximate the
gradient with **finite differences** on each integer-valued LACE
component. We then walk epsilon in increments to find the smallest
L-infinity perturbation that flips the recommended action.

Pure-Python deterministic. Validates the system's robustness to
small input perturbations - a critical assurance for any clinical
classifier deployed in adversarial settings.
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field


_Predictor = Callable[[dict[str, float]], float]


class FGSMReport(BaseModel):
    original_features: dict[str, float]
    original_score: float
    original_class: int
    epsilon_min_to_flip: float | None
    adversarial_features: dict[str, float]
    adversarial_score: float
    adversarial_class: int
    flipped: bool
    finite_difference_gradient: dict[str, float]
    sign_perturbation: dict[str, int]
    rationale: str


def _numeric_gradient(
    features: dict[str, float],
    predictor: _Predictor,
    h: float = 1.0,
) -> dict[str, float]:
    """Forward-difference numerical gradient (one-sided).

    Returns dF/dx_k approximated as
        (predictor(features + h e_k) - predictor(features)) / h
    """
    base = float(predictor(features))
    grads: dict[str, float] = {}
    for k in features:
        bumped = dict(features)
        bumped[k] = float(features[k]) + h
        grads[k] = (float(predictor(bumped)) - base) / h
    return grads


def _sign(x: float) -> int:
    if x > 1e-12:
        return 1
    if x < -1e-12:
        return -1
    return 0


def fgsm_attack(
    *,
    features: dict[str, float],
    predictor: _Predictor,
    decision_threshold: float = 0.20,
    epsilon_max: float = 5.0,
    epsilon_step: float = 0.5,
    feature_min: dict[str, float] | None = None,
    feature_max: dict[str, float] | None = None,
    targeted_increase: bool = True,
) -> FGSMReport:
    """Run an FGSM walk and return the smallest epsilon (L-infinity)
    that flips the prediction class.

    Args:
        features: clean input feature dict.
        predictor: callable returning a probability in [0, 1].
        decision_threshold: class boundary in the probability space.
        epsilon_max: stop searching beyond this perturbation.
        epsilon_step: step size for the epsilon walk.
        feature_min / feature_max: clip the perturbed input to a
            plausible domain (e.g., LACE components in [0, 7] etc.).
        targeted_increase: True -> push the score up; False -> push
            it down. The sign of the gradient is multiplied by +/-1
            accordingly.
    """
    if not features:
        raise ValueError("features cannot be empty")
    if epsilon_max <= 0 or epsilon_step <= 0:
        raise ValueError(
            "epsilon_max and epsilon_step must be > 0")
    feature_min = feature_min or {}
    feature_max = feature_max or {}

    base_score = float(predictor(features))
    base_class = int(base_score >= decision_threshold)
    grad = _numeric_gradient(features, predictor)
    direction = 1 if targeted_increase else -1
    sign_perturb = {
        k: direction * _sign(grad[k]) for k in grad
    }

    flipped = False
    eps_min: float | None = None
    adv_features = dict(features)
    adv_score = base_score
    adv_class = base_class

    eps = epsilon_step
    while eps <= epsilon_max + 1e-9:
        candidate = {}
        for k in features:
            v = float(features[k]) + eps * sign_perturb.get(k, 0)
            if k in feature_min:
                v = max(v, feature_min[k])
            if k in feature_max:
                v = min(v, feature_max[k])
            candidate[k] = v
        score = float(predictor(candidate))
        cls = int(score >= decision_threshold)
        if cls != base_class:
            flipped = True
            eps_min = round(eps, 6)
            adv_features = candidate
            adv_score = score
            adv_class = cls
            break
        eps += epsilon_step

    rationale = (
        f"FGSM with finite-difference gradient on {len(features)} "
        f"features; original score {base_score:.3f} -> "
        f"{adv_score:.3f}, "
        f"{'flipped at L-infinity epsilon=' + str(eps_min) if flipped else 'no flip within budget'}."
    )
    return FGSMReport(
        original_features={k: round(float(v), 6)
                           for k, v in features.items()},
        original_score=round(base_score, 6),
        original_class=base_class,
        epsilon_min_to_flip=eps_min,
        adversarial_features={k: round(float(v), 6)
                              for k, v in adv_features.items()},
        adversarial_score=round(adv_score, 6),
        adversarial_class=adv_class,
        flipped=flipped,
        finite_difference_gradient={
            k: round(v, 6) for k, v in grad.items()
        },
        sign_perturbation=sign_perturb,
        rationale=rationale,
    )


# ─────────────────────────────────────────────────────────────────────
# Robustness sweep
# ─────────────────────────────────────────────────────────────────────


class RobustnessReport(BaseModel):
    n_instances: int = Field(ge=0)
    pct_robust_at_eps_1: float = Field(ge=0.0, le=1.0)
    pct_robust_at_eps_2: float = Field(ge=0.0, le=1.0)
    pct_robust_at_eps_3: float = Field(ge=0.0, le=1.0)
    avg_eps_min_to_flip: float
    rationale: str


def adversarial_robustness_sweep(
    *,
    instances: list[dict[str, float]],
    predictor: _Predictor,
    decision_threshold: float = 0.20,
    epsilon_max: float = 5.0,
    epsilon_step: float = 0.5,
    feature_min: dict[str, float] | None = None,
    feature_max: dict[str, float] | None = None,
) -> RobustnessReport:
    """Run FGSM on each instance and aggregate the robustness curve.

    Reports the fraction of instances that resist a perturbation of
    L-infinity epsilon <= 1.0 / 2.0 / 3.0, and the average epsilon
    needed to flip when the attack succeeds.
    """
    if not instances:
        raise ValueError("instances cannot be empty")
    eps_to_flip: list[float] = []
    n_robust_1 = n_robust_2 = n_robust_3 = 0
    n_unflippable = 0
    for inst in instances:
        rep = fgsm_attack(
            features=inst, predictor=predictor,
            decision_threshold=decision_threshold,
            epsilon_max=epsilon_max,
            epsilon_step=epsilon_step,
            feature_min=feature_min, feature_max=feature_max,
        )
        if rep.flipped and rep.epsilon_min_to_flip is not None:
            eps_to_flip.append(rep.epsilon_min_to_flip)
            if rep.epsilon_min_to_flip > 1.0:
                n_robust_1 += 1
            if rep.epsilon_min_to_flip > 2.0:
                n_robust_2 += 1
            if rep.epsilon_min_to_flip > 3.0:
                n_robust_3 += 1
        else:
            n_unflippable += 1
            n_robust_1 += 1
            n_robust_2 += 1
            n_robust_3 += 1
    n = len(instances)
    avg_eps = (
        sum(eps_to_flip) / len(eps_to_flip)
        if eps_to_flip else float("nan")
    )
    return RobustnessReport(
        n_instances=n,
        pct_robust_at_eps_1=round(n_robust_1 / n, 6),
        pct_robust_at_eps_2=round(n_robust_2 / n, 6),
        pct_robust_at_eps_3=round(n_robust_3 / n, 6),
        avg_eps_min_to_flip=(
            round(avg_eps, 4) if eps_to_flip else 0.0
        ),
        rationale=(
            f"Sweep of {n} instances; "
            f"{n_unflippable} unflippable within budget; "
            f"{n_robust_1}/{n} robust at L-inf epsilon <= 1.0; "
            f"average flipping epsilon {avg_eps:.2f}."
        ),
    )
