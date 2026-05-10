"""PATIENT-4 -- Multi-channel notification formatter.

Pure-formatting layer over a `DischargeCounseling` (or
`TranslatedDischargeCounseling`) that produces three delivery-ready
renderings:

  - sms             : 160-char chunks, plain-text, Unicode-safe
  - email_html      : minimal accessible HTML (headings, lists, table-free)
  - print_markdown  : plain Markdown suitable for PDF rendering

No LLM involved -- this is a deterministic adapter. Same content surface
for every channel; only the chunking + markup differ.
"""

from __future__ import annotations

import html
import re
from typing import Iterable

from shared.schemas import (
    CounselingSection,
    DischargeCounseling,
    FormattedNotificationBundle,
    FormattedNotificationChannel,
    TranslatedDischargeCounseling,
)


# ─────────────────────── Source coercion ───────────────────────

def _coerce_source(source) -> tuple[list[CounselingSection], str]:
    """Return (sections, disclaimer). Accepts either DischargeCounseling
    or TranslatedDischargeCounseling (object or dict)."""
    if isinstance(source, dict):
        if "target_locale" in source:
            obj = TranslatedDischargeCounseling.model_validate(source)
        else:
            obj = DischargeCounseling.model_validate(source)
    else:
        obj = source
    return list(obj.sections), obj.disclaimer


# ─────────────────────── SMS chunking ───────────────────────

_SMS_LIMIT = 160


def _flatten_sections(sections: Iterable[CounselingSection],
                        disclaimer: str) -> str:
    """Concatenate sections into a single plain-text body for SMS chunking."""
    parts: list[str] = []
    for s in sections:
        parts.append(s.title.upper())
        if s.plain_text:
            parts.append(s.plain_text)
        for b in s.bullets:
            parts.append("- " + b)
        parts.append("")
    parts.append("--")
    parts.append(disclaimer)
    return "\n".join(parts)


def _sms_chunks(body: str, limit: int = _SMS_LIMIT) -> list[str]:
    """Split body into ≤limit-char chunks ending with paginators (1/3, 2/3, 3/3).

    Splits on sentence boundaries when possible; falls back to whitespace
    when not. Every chunk is guaranteed ≤ limit chars including the
    pagination suffix.
    """
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []

    paginator_reserve = 8   # space for " (NN/NN)"
    soft_limit = max(1, limit - paginator_reserve)

    raw_chunks: list[str] = []
    cursor = 0
    while cursor < len(body):
        end = min(cursor + soft_limit, len(body))
        if end < len(body):
            # Backtrack to the last sentence boundary or whitespace
            for sep in (". ", "! ", "? ", "\n", " "):
                idx = body.rfind(sep, cursor, end)
                if idx > cursor + 20:
                    end = idx + len(sep)
                    break
        raw_chunks.append(body[cursor:end].strip())
        cursor = end

    n = len(raw_chunks)
    out: list[str] = []
    for i, ch in enumerate(raw_chunks, start=1):
        suffix = f" ({i}/{n})"
        out.append((ch + suffix)[:limit])
    return out


# ─────────────────────── Email HTML rendering ───────────────────────

def _email_html(sections: list[CounselingSection],
                  disclaimer: str) -> str:
    """Build a minimal accessible HTML email body -- no inline styles, no JS,
    no remote fetches. Designed for the lowest-common-denominator email
    clients (Outlook, plain Gmail, Apple Mail)."""
    out: list[str] = ['<div role="article" lang="en">']
    out.append('<h1>Your discharge summary</h1>')
    for sec in sections:
        out.append(f'<h2>{html.escape(sec.title)}</h2>')
        if sec.plain_text:
            out.append(f'<p>{html.escape(sec.plain_text)}</p>')
        if sec.bullets:
            out.append("<ul>")
            for b in sec.bullets:
                out.append(f'  <li>{html.escape(b)}</li>')
            out.append("</ul>")
    out.append("<hr>")
    out.append(f'<p><em>{html.escape(disclaimer)}</em></p>')
    out.append("</div>")
    return "\n".join(out)


# ─────────────────────── Print Markdown rendering ───────────────────────

def _print_markdown(sections: list[CounselingSection],
                      disclaimer: str) -> str:
    out: list[str] = ["# Your discharge summary", ""]
    for sec in sections:
        out.append(f"## {sec.title}")
        if sec.plain_text:
            out.append("")
            out.append(sec.plain_text)
        if sec.bullets:
            out.append("")
            for b in sec.bullets:
                out.append(f"- {b}")
        out.append("")
    out.append("---")
    out.append("")
    out.append(f"*{disclaimer}*")
    return "\n".join(out)


# ─────────────────────── Public API ───────────────────────

_VALID_CHANNELS = ("sms", "email_html", "print_markdown")


def format_discharge_notifications(
    source: DischargeCounseling | TranslatedDischargeCounseling | dict,
    channels: list[str] | None = None,
    sms_chunk_limit: int = _SMS_LIMIT,
) -> FormattedNotificationBundle:
    """Render the discharge counseling for one or more delivery channels.

    Args:
        source: a DischargeCounseling, TranslatedDischargeCounseling, or
            dict equivalent.
        channels: subset of ('sms', 'email_html', 'print_markdown'); default
            all three.
        sms_chunk_limit: characters per SMS chunk (default 160).

    Returns:
        FormattedNotificationBundle with per-channel renderings.
    """
    if sms_chunk_limit < 40 or sms_chunk_limit > 1600:
        raise ValueError("sms_chunk_limit must be in [40, 1600].")
    if channels is None:
        channels = list(_VALID_CHANNELS)
    invalid = [c for c in channels if c not in _VALID_CHANNELS]
    if invalid:
        raise ValueError(
            f"unsupported channels: {invalid}. "
            f"Valid: {list(_VALID_CHANNELS)}.")

    sections, disclaimer = _coerce_source(source)

    out_channels: list[FormattedNotificationChannel] = []

    if "sms" in channels:
        body = _flatten_sections(sections, disclaimer)
        chunks = _sms_chunks(body, limit=sms_chunk_limit)
        out_channels.append(FormattedNotificationChannel(
            channel="sms",
            chunks=chunks,
            n_chunks=len(chunks),
            total_characters=sum(len(c) for c in chunks),
        ))

    if "email_html" in channels:
        body = _email_html(sections, disclaimer)
        out_channels.append(FormattedNotificationChannel(
            channel="email_html",
            chunks=[body],
            n_chunks=1,
            total_characters=len(body),
        ))

    if "print_markdown" in channels:
        body = _print_markdown(sections, disclaimer)
        out_channels.append(FormattedNotificationChannel(
            channel="print_markdown",
            chunks=[body],
            n_chunks=1,
            total_characters=len(body),
        ))

    return FormattedNotificationBundle(
        sections_formatted=len(sections),
        channels=out_channels,
    )
