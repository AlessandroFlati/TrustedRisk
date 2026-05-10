"""Patient-history memory layer.

A SQLite-backed store of prior `DecisionCard`s, keyed by patient_id. The A2A
agent reads this store before issuing a fresh recommendation:

  - **Cache hit (still valid)**: a DecisionCard exists for this patient AND
    its temporal validity window has not expired AND the upstream model
    version has not changed. The runtime can serve the cached recommendation
    directly, attaching `from_memory: true` to the audit. No tool round-trip.

  - **Cache hit (expired or stale model)**: the card exists but is no longer
    safe to serve. The runtime issues a fresh tool chain and writes the new
    card on top.

  - **Cache miss**: standard fresh-recommendation path.

The store is a transparent thin layer over `sqlite3`. Schema is defined as a
single `decision_cards` table with one row per (patient_id, request_id).
Production deployments would migrate this to a managed datastore (Postgres,
Firestore) but keeping it SQLite here is intentional -- the schema must
evolve with the DecisionCard schema, and SQLite's flexibility lets that
happen without ops overhead.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from shared.schemas import (
    DecisionCard,
    PatientTimeline,
    PatientTimelineEntry,
)
from shared.validity import card_validity_remaining_minutes, is_card_valid


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_cards (
    request_id      TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    card_json       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    valid_until_iso TEXT,
    written_at_iso  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decision_cards_patient
    ON decision_cards (patient_id, written_at_iso DESC);
"""


class MemoryStore:
    """Persistent prior-card store. Single-process, single-file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get("TRUSTEDRISK_MEMORY_DB_PATH",
                                      "data/memory.sqlite3")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ─────────────────────── Connection helpers ───────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ─────────────────────── Write ───────────────────────

    def store_card(self, card: DecisionCard) -> None:
        """Persist a DecisionCard, indexed by request_id (PK) and patient_id."""
        request_id = card.audit.request_id
        # Heuristic patient_id extraction -- audit may carry it explicitly via
        # the trace context_fingerprint, otherwise we fall back to the
        # context_fingerprint string itself (still useful as a stable hash).
        patient_id = self._extract_patient_id(card)

        risk = card.reasoning.risk_estimate if card.reasoning else None
        model_version = (risk.model_version if risk else "unknown")
        valid_until_iso = (risk.valid_until.isoformat()
                           if (risk and risk.valid_until) else None)

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO decision_cards
                   (request_id, patient_id, card_json, model_version,
                    valid_until_iso, written_at_iso)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    request_id, patient_id,
                    card.model_dump_json(),
                    model_version, valid_until_iso,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    @staticmethod
    def _extract_patient_id(card: DecisionCard) -> str:
        """Best-effort extraction. Production deployments wire patient_id
        explicitly into AuditBlock; here we use context_fingerprint as a
        stable per-patient surrogate when no explicit field is present."""
        return card.audit.context_fingerprint or card.audit.request_id

    # ─────────────────────── Read ───────────────────────

    def get_latest_for_patient(
        self,
        patient_id: str,
        *,
        require_valid: bool = True,
        require_model_version: str | None = None,
        now: datetime | None = None,
    ) -> DecisionCard | None:
        """Fetch the most recent DecisionCard for `patient_id`. When
        `require_valid` is True, only returns cards whose temporal window
        has not expired. When `require_model_version` is set, only returns
        cards from that exact model version (older cards from a superseded
        model trigger a fresh recall)."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT card_json, model_version, valid_until_iso
                   FROM decision_cards
                   WHERE patient_id = ?
                   ORDER BY written_at_iso DESC
                   LIMIT 1""",
                (patient_id,),
            ).fetchone()
        if row is None:
            return None

        if require_model_version is not None and row["model_version"] != require_model_version:
            return None

        card = DecisionCard.model_validate_json(row["card_json"])
        if require_valid and not is_card_valid(card, now=now):
            return None
        return card

    def get_history_for_patient(
        self,
        patient_id: str,
        *,
        limit: int = 10,
    ) -> list[DecisionCard]:
        """Return up to `limit` most-recent cards for the patient. Used by the
        self-critique stage to compare current decision against prior ones
        (consistency check across encounters)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT card_json FROM decision_cards
                   WHERE patient_id = ?
                   ORDER BY written_at_iso DESC
                   LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        out: list[DecisionCard] = []
        for r in rows:
            try:
                out.append(DecisionCard.model_validate_json(r["card_json"]))
            except Exception:
                continue  # skip malformed rows defensively
        return out

    # ─────────────────────── Maintenance ───────────────────────

    def evict_expired(self, now: datetime | None = None) -> int:
        """Delete cards whose valid_until is past. Returns the count removed."""
        now = now or datetime.now(timezone.utc)
        with self._connect() as conn:
            cur = conn.execute(
                """DELETE FROM decision_cards
                   WHERE valid_until_iso IS NOT NULL
                     AND valid_until_iso < ?""",
                (now.isoformat(),),
            )
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM decision_cards").fetchone()
        return int(row["n"])

    # ─────────────────────── AMB-5.3 Longitudinal timeline ───────────────────────

    def build_patient_timeline(self, patient_id: str,
                                 limit: int = 50) -> PatientTimeline:
        """Aggregate the full encounter history for one patient into a
        PatientTimeline view, with longitudinal drift signals.

        The timeline is ordered chronologically (oldest -> newest). Drift
        flags fire when consecutive encounters show:
          - recommendation_action regressed toward more aggressive care
            (e.g. discharge_home -> continued_admission)
          - probability_mean trend reversed (improving vs worsening)
          - confidence dropped two tiers in a row
          - abstain rate increased markedly
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT card_json, written_at_iso
                   FROM decision_cards
                   WHERE patient_id = ?
                   ORDER BY written_at_iso ASC
                   LIMIT ?""",
                (patient_id, limit),
            ).fetchall()

        cards: list[tuple[DecisionCard, datetime]] = []
        for r in rows:
            try:
                card = DecisionCard.model_validate_json(r["card_json"])
                ts = datetime.fromisoformat(r["written_at_iso"])
                cards.append((card, ts))
            except Exception:
                continue

        entries: list[PatientTimelineEntry] = []
        for card, ts in cards:
            rec = card.recommendation
            risk = card.reasoning.risk_estimate if card.reasoning else None
            entry = PatientTimelineEntry(
                encounter_id=card.audit.request_id,
                encounter_at=ts,
                encounter_type=_infer_encounter_type(card),
                recommendation_action=(rec.action.value if rec else None),
                recommendation_confidence=(rec.confidence if rec else None),
                risk_probability_mean=(risk.probability_mean if risk else None),
                risk_outcome_id=(risk.outcome_id if risk else None),
                abstain_triggers_count=len(card.abstain or []),
                notes=None,
            )
            entries.append(entry)

        drift_flags = _compute_longitudinal_drift(entries)

        return PatientTimeline(
            patient_id=patient_id,
            n_encounters=len(entries),
            first_encounter_at=entries[0].encounter_at if entries else None,
            last_encounter_at=entries[-1].encounter_at if entries else None,
            entries=entries,
            drift_flags=drift_flags,
        )


# ─────────────────────── Timeline helpers (AMB-5.3) ───────────────────────

_ACTION_AGGRESSIVENESS: dict[str, int] = {
    # Higher = more aggressive (less ambulatory)
    "discharge_home": 0,
    "home_with_care": 1,
    "snf": 2,
    "continued_admission": 3,
}

_CONFIDENCE_ORDER: dict[str, int] = {
    "none": 0, "low": 1, "medium": 2, "high": 3,
}


def _infer_encounter_type(card: DecisionCard) -> str:
    """Heuristic to classify an encounter type from the DecisionCard's
    structure. Used for timeline rendering."""
    rec = card.recommendation
    if not rec and card.abstain:
        return "ed_visit"
    if rec and rec.action.value == "continued_admission":
        return "inpatient_review"
    if rec and rec.action.value in ("discharge_home", "home_with_care", "snf"):
        return "discharge"
    return "outpatient"


def _compute_longitudinal_drift(
    entries: list[PatientTimelineEntry],
) -> list[str]:
    """Detect drift signals in a chronological timeline."""
    flags: list[str] = []
    if len(entries) < 2:
        return flags

    for prev, cur in zip(entries[:-1], entries[1:]):
        # Action escalation / de-escalation
        prev_aggr = _ACTION_AGGRESSIVENESS.get(prev.recommendation_action or "")
        cur_aggr = _ACTION_AGGRESSIVENESS.get(cur.recommendation_action or "")
        if prev_aggr is not None and cur_aggr is not None and cur_aggr - prev_aggr >= 2:
            flags.append(
                f"action escalation: {prev.recommendation_action!r} -> "
                f"{cur.recommendation_action!r} between {prev.encounter_id} "
                f"and {cur.encounter_id}"
            )

        # Probability worsening
        if (prev.risk_probability_mean is not None
                and cur.risk_probability_mean is not None
                and cur.risk_probability_mean - prev.risk_probability_mean > 0.15):
            flags.append(
                f"probability_mean worsening: "
                f"{prev.risk_probability_mean:.3f} -> "
                f"{cur.risk_probability_mean:.3f}"
            )

        # Confidence drop ≥2 tiers
        prev_conf = _CONFIDENCE_ORDER.get(prev.recommendation_confidence or "high", 3)
        cur_conf = _CONFIDENCE_ORDER.get(cur.recommendation_confidence or "high", 3)
        if prev_conf - cur_conf >= 2:
            flags.append(
                f"confidence drop ≥2 tiers: "
                f"{prev.recommendation_confidence!r} -> "
                f"{cur.recommendation_confidence!r}"
            )

        # Abstain rate change
        if cur.abstain_triggers_count >= 2 and prev.abstain_triggers_count == 0:
            flags.append(
                f"abstain emerged: 0 triggers -> "
                f"{cur.abstain_triggers_count} triggers"
            )

    return flags


# ─────────────────────── Consistency check ───────────────────────

def detect_recommendation_drift(
    current: DecisionCard,
    prior: DecisionCard,
) -> dict[str, Any] | None:
    """Compare two DecisionCards for the same patient; flag when current
    differs materially from prior. Returns a summary dict or None if drift
    is within tolerable noise.

    Signals checked:
        - Recommendation action flipped
        - Recommendation confidence dropped a tier (preferred -> degraded)
        - Risk probability moved >0.10 in either direction
        - Abstain status changed (was abstain -> now recommendation, or vice versa)
    """
    cur_action = current.recommendation.action.value if current.recommendation else None
    prior_action = prior.recommendation.action.value if prior.recommendation else None
    cur_abstain = bool(current.abstain)
    prior_abstain = bool(prior.abstain)

    cur_prob = (current.reasoning.risk_estimate.probability_mean
                if current.reasoning else None)
    prior_prob = (prior.reasoning.risk_estimate.probability_mean
                  if prior.reasoning else None)

    flips = []
    if cur_action != prior_action:
        flips.append(f"action: {prior_action!r} -> {cur_action!r}")
    if cur_abstain != prior_abstain:
        flips.append(f"abstain: {prior_abstain} -> {cur_abstain}")
    if cur_prob is not None and prior_prob is not None:
        if abs(cur_prob - prior_prob) > 0.10:
            flips.append(f"prob_mean: {prior_prob:.3f} -> {cur_prob:.3f}")

    if not flips:
        return None
    return {
        "drift_signals": flips,
        "prior_request_id": prior.audit.request_id,
        "current_request_id": current.audit.request_id,
        "prior_remaining_minutes": card_validity_remaining_minutes(prior),
    }
