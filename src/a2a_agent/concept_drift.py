"""Phase 17.AT - Concept-drift online detection.

Two canonical online drift detectors for streaming predictions:

  - **ADWIN** (Bifet & Gavaldà 2007) - ADaptive WINdowing. Maintains a
    sliding window of observations and detects a change-point when
    the means of two sub-windows diverge beyond a Hoeffding bound.
    Useful for tracking continuous loss / probability streams.

  - **DDM** (Drift Detection Method, Gama 2004) - tracks the running
    error rate + standard deviation on a binary mis-classification
    stream and raises *warning* / *drift* alarms when the error
    deviates from the historical minimum by 2 sigma / 3 sigma.

Pure-Python deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# ADWIN
# ─────────────────────────────────────────────────────────────────────


class ADWINReport(BaseModel):
    n_samples_seen: int = Field(ge=0)
    window_size: int = Field(ge=0)
    window_mean: float
    drift_detected_at: list[int]
    rationale: str


class ADWIN:
    """Bifet & Gavaldà 2007 ADWIN (simplified equal-split variant).

    On every new observation, compare the means of the first half +
    second half of the window. If the gap exceeds the Hoeffding
    bound

        epsilon_cut = sqrt( (1/(2 m)) * log(4/delta) )

    where m = harmonic mean of half-window sizes, drop the older
    half and record a drift event.
    """

    def __init__(self, *, delta: float = 0.002,
                 max_window: int = 5000) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if max_window < 4:
            raise ValueError("max_window must be >= 4")
        self._delta = delta
        self._max_window = max_window
        self._window: deque[float] = deque(maxlen=max_window)
        self._n_seen = 0
        self._drifts: list[int] = []

    def add(self, value: float) -> bool:
        """Push a new observation; return True if a drift fires."""
        self._n_seen += 1
        self._window.append(float(value))
        return self._check()

    def _check(self) -> bool:
        n = len(self._window)
        if n < 4:
            return False
        # Try every split point and keep the earliest qualifying cut
        values = list(self._window)
        total = sum(values)
        running_left = 0.0
        for split in range(2, n - 1):
            running_left += values[split - 1]
            mean_left = running_left / split
            mean_right = (total - running_left) / (n - split)
            m = (1.0 / split + 1.0 / (n - split))
            try:
                eps = math.sqrt(0.5 * m * math.log(4.0 / self._delta))
            except ValueError:
                continue
            if abs(mean_left - mean_right) > eps:
                # Drift: drop the older half
                for _ in range(split):
                    self._window.popleft()
                self._drifts.append(self._n_seen)
                return True
        return False

    def report(self) -> ADWINReport:
        n = len(self._window)
        mean = sum(self._window) / n if n > 0 else 0.0
        return ADWINReport(
            n_samples_seen=self._n_seen,
            window_size=n,
            window_mean=round(mean, 6),
            drift_detected_at=list(self._drifts),
            rationale=(
                f"ADWIN delta={self._delta}; {len(self._drifts)} "
                f"drift events; current window n={n}, mean="
                f"{mean:.4f}."
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# DDM
# ─────────────────────────────────────────────────────────────────────


_DDMState = Literal["in_control", "warning", "drift"]


class DDMReport(BaseModel):
    n_samples_seen: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    std: float = Field(ge=0.0)
    p_min_plus_2s: float
    p_min_plus_3s: float
    state: _DDMState
    drift_indices: list[int]
    warning_indices: list[int]
    rationale: str


class DDM:
    """Gama et al. 2004 Drift Detection Method.

    State machine:
      - in_control: error <= p_min + 2 sigma_min
      - warning:    p_min + 2 sigma_min < error <= p_min + 3 sigma_min
      - drift:      error > p_min + 3 sigma_min  (-> reset stats)
    """

    def __init__(self, *, min_n_before_check: int = 30) -> None:
        if min_n_before_check < 5:
            raise ValueError("min_n_before_check must be >= 5")
        self._min_n = min_n_before_check
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._errors = 0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._state: _DDMState = "in_control"
        self._drifts: list[int] = []
        self._warnings: list[int] = []

    def add(self, error: int) -> _DDMState:
        """Push a binary error (1 = misclassified, 0 = correct);
        return the post-update state."""
        if error not in (0, 1):
            raise ValueError("error must be 0 or 1")
        self._n += 1
        self._errors += error
        p = self._errors / self._n
        s = math.sqrt(p * (1 - p) / self._n) if self._n > 0 else 0.0
        if self._n < self._min_n:
            self._state = "in_control"
            return self._state
        if (p + s) < (self._p_min + self._s_min):
            self._p_min = p
            self._s_min = s
        thr_warn = self._p_min + 2 * self._s_min
        thr_drift = self._p_min + 3 * self._s_min
        if p + s > thr_drift:
            self._drifts.append(self._n)
            self.reset()
            # reset() wipes self._state; the caller still needs to
            # see this transition as a drift.
            return "drift"
        elif p + s > thr_warn:
            self._state = "warning"
            self._warnings.append(self._n)
        else:
            self._state = "in_control"
        return self._state

    def report(self) -> DDMReport:
        p = (self._errors / self._n) if self._n > 0 else 0.0
        s = (
            math.sqrt(p * (1 - p) / self._n)
            if self._n > 0 else 0.0
        )
        return DDMReport(
            n_samples_seen=self._n,
            error_rate=round(p, 6),
            std=round(s, 6),
            p_min_plus_2s=round(
                self._p_min + 2 * self._s_min, 6
            ) if self._p_min < float("inf") else 0.0,
            p_min_plus_3s=round(
                self._p_min + 3 * self._s_min, 6
            ) if self._p_min < float("inf") else 0.0,
            state=self._state,
            drift_indices=list(self._drifts),
            warning_indices=list(self._warnings),
            rationale=(
                f"DDM: n={self._n}, p={p:.4f}; "
                f"{len(self._drifts)} drift / "
                f"{len(self._warnings)} warning events; current "
                f"state `{self._state}`."
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# Streaming convenience
# ─────────────────────────────────────────────────────────────────────


def detect_drifts_adwin(
    *, stream: Iterable[float], delta: float = 0.002,
) -> list[int]:
    """Convenience: feed ``stream`` to ADWIN + return drift indices."""
    a = ADWIN(delta=delta)
    drifts: list[int] = []
    for x in stream:
        if a.add(x):
            drifts.append(a._n_seen)
    return drifts


def detect_drifts_ddm(
    *, errors: Iterable[int], min_n_before_check: int = 30,
) -> list[int]:
    """Stream ``errors`` through DDM; return per-stream-position
    drift indices. Captures every drift even though the detector
    resets its internal state on each one."""
    d = DDM(min_n_before_check=min_n_before_check)
    drifts: list[int] = []
    pos = 0
    for e in errors:
        pos += 1
        state = d.add(e)
        if state == "drift":
            drifts.append(pos)
    return drifts
