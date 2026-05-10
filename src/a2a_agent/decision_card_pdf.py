"""Phase 14.10 N1 -- PDF export of DecisionCard.

Renders a clinical-grade printable PDF without third-party deps. We
hand-build a minimal PDF 1.4 document using only stdlib `zlib` for
compression -- keeps the runtime dependency-free in production AND
the byte output deterministic for tests.

The output is a 1-page A4 document with:
  - Header: title + patient stub + timestamp
  - Recommendation block (recommended_action)
  - Risk estimate (point + CI95 / conformal interval)
  - Contributing factors table
  - Cited evidence list
  - Footer: deterministic hash for audit trail

This is intentionally minimal -- for production-grade typography one
would swap to ReportLab or wkhtmltopdf. The deterministic-stream
shape is the test contract we ship.
"""

from __future__ import annotations

import hashlib
import zlib
from datetime import datetime, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────


def _escape_pdf_text(text: str) -> str:
    """Escape parens / backslashes for PDF string literals."""
    return (
        text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
    )


def _build_content_stream(lines: list[tuple[float, float, str, int]]) -> bytes:
    """Build a single content stream from (x, y, text, font_size) lines.
    Font is the standard built-in Helvetica (no font subset embedding)."""
    parts: list[str] = ["BT"]
    for x, y, text, size in lines:
        parts.append(f"/F1 {size} Tf")
        parts.append(f"1 0 0 1 {x:.2f} {y:.2f} Tm")
        parts.append(f"({_escape_pdf_text(text)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("utf-8")


def _build_pdf(lines: list[tuple[float, float, str, int]]) -> bytes:
    """Build a complete A4 1-page PDF from the prepared text lines."""
    content = _build_content_stream(lines)
    compressed = zlib.compress(content)

    objects: list[bytes] = []

    def _add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    # 1: Catalog
    catalog_idx = _add(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    pages_idx = _add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    page_idx = _add(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    # 4: Content stream
    content_obj = (
        f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n"
        f"stream\n"
    ).encode("ascii") + compressed + b"\nendstream"
    content_idx = _add(content_obj)
    # 5: Font (Helvetica built-in)
    font_idx = _add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    # Assemble
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += b"trailer\n"
    out += (
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    ).encode("ascii")
    out += b"startxref\n"
    out += f"{xref_offset}\n".encode("ascii")
    out += b"%%EOF\n"
    return bytes(out)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def render_decision_card_pdf(
    *,
    title: str = "TrustedRisk DecisionCard",
    patient_label: str | None = None,
    recommended_action: str | None = None,
    risk_point_estimate: float | None = None,
    risk_ci95: tuple[float, float] | None = None,
    contributing_factors: list[dict[str, Any]] | None = None,
    cited_evidence: list[str] | None = None,
    timestamp_iso: str | None = None,
) -> bytes:
    """Render a 1-page A4 PDF of a DecisionCard.

    Returns:
        bytes -- the complete PDF document. Deterministic given the
        same inputs (same byte sequence, useful for tests).
    """
    timestamp_iso = timestamp_iso or datetime.now(
        timezone.utc,
    ).isoformat()

    # Assemble text lines (PDF y-axis: bottom-up; A4 = 842 pt high)
    lines: list[tuple[float, float, str, int]] = []
    y = 800
    lines.append((40, y, title, 18))
    y -= 28
    lines.append((40, y, f"Generated: {timestamp_iso}", 9))
    y -= 14
    if patient_label:
        lines.append((40, y, f"Patient: {patient_label}", 10))
        y -= 18

    y -= 8
    lines.append((40, y, "Recommendation", 12))
    y -= 16
    lines.append((50, y, recommended_action or "(unspecified)", 10))
    y -= 22

    if risk_point_estimate is not None:
        lines.append((40, y, "Risk estimate", 12))
        y -= 16
        lines.append((50, y,
                          f"Point: {risk_point_estimate:.3f}",
                          10))
        y -= 14
        if risk_ci95 is not None:
            lines.append((50, y,
                              f"CI95: [{risk_ci95[0]:.3f}, "
                              f"{risk_ci95[1]:.3f}]", 10))
            y -= 14
        y -= 8

    if contributing_factors:
        lines.append((40, y, "Contributing factors", 12))
        y -= 16
        for f in contributing_factors[:8]:
            name = f.get("name", "factor")
            pts = f.get("lace_points", 0)
            wt = f.get("weight", 0.0)
            lines.append((50, y,
                              f"• {name}: {pts} pts (weight {wt:.2f})",
                              9))
            y -= 12
        y -= 8

    if cited_evidence:
        lines.append((40, y, "Cited evidence", 12))
        y -= 16
        for ev in cited_evidence[:10]:
            lines.append((50, y, f"• {str(ev)[:80]}", 9))
            y -= 12

    # Footer with audit hash
    payload = (
        f"{title}|{patient_label}|{recommended_action}|"
        f"{risk_point_estimate}|{risk_ci95}|"
        f"{len(contributing_factors or [])}|"
        f"{len(cited_evidence or [])}|"
        f"{timestamp_iso}"
    )
    audit_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    lines.append((40, 40, f"audit-hash: {audit_hash}", 7))

    return _build_pdf(lines)
