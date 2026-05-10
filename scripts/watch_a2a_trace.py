"""Tail logs/a2a_trace.jsonl in real time, pretty-printing each request/response.

Use this while testing prompts in Prompt Opinion: launch BEFORE clicking
in PO and the script renders each A2A round-trip as it lands.

Known-fail-pattern detection: every entry is scanned for failure modes the
operator should not miss in a wall of JSON. When a pattern matches, a red
banner prints ABOVE the response render, naming the failure class and the
specific step / artifact / message that triggered it. Currently detected:

  - FHIR-CONTEXT LEAK: a tool inside a workflow step raised
    `LookupError: <ContextVar name='fhir_ctx' ...>`. The PO FHIR-context
    metadata never reached the tool's bound ContextVar; the most likely
    cause is that the federation process is on an old build that predates
    the A2A backend bind seam (commit 1eb7fd9 / v0.9.0).
  - SHARP REJECTION: a SHARP middleware path returned `missing_fhir_context`
    or 403; FHIR headers absent at the HTTP layer.
  - OAUTH FAIL: the orchestrator endpoint returned `auth_required`,
    `invalid_token`, `invalid_grant`, or 401.
  - AGENT EXCEPTION: a tool callable raised an unexpected exception that
    bubbled up to the JSON-RPC error code -32000.
  - DISCOVERY FALLBACK: the engine returned the discovery catalog when a
    real workflow was probably expected (LLM down or prompt too vague).

Usage (PowerShell from repo root):
    .venv\\Scripts\\python.exe scripts\\watch_a2a_trace.py

Optional flags:
    --tail N        Replay the last N entries before tailing (default 0).
    --no-payload    Print only the routing summary, drop the full artifact JSON.
    --filter SLUG   Print only entries whose response touches this slug
                    (e.g. trustedrisk-orchestrator, trustedrisk-appeals).
    --errors-only   Print only entries where at least one failure pattern
                    matched (useful for sustained debugging sessions).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "logs" / "a2a_trace.jsonl"


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


def _green(s: str) -> str:
    return _color(s, "32")


def _yellow(s: str) -> str:
    return _color(s, "33")


def _red(s: str) -> str:
    return _color(s, "31")


def _blue(s: str) -> str:
    return _color(s, "34")


def _grey(s: str) -> str:
    return _color(s, "90")


def _bold(s: str) -> str:
    return _color(s, "1")


def _bg_red(s: str) -> str:
    """Bold white text on red background — used for the fail banner."""
    return f"\033[1;37;41m{s}\033[0m"


# ─────────────────────── Failure-pattern detection ───────────────────────

# (label, regex, hint) — the regex matches against the JSON-serialised
# response payload. The hint is printed under the banner to point the
# operator at the most likely root cause.
_FAIL_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "FHIR-CONTEXT LEAK",
        re.compile(r"LookupError[^\"]*fhir_ctx", re.IGNORECASE),
        "A tool ran without FHIR-context bound. "
        "Check that the federation is on v0.9.0+ (commit 1eb7fd9 added the "
        "A2A backend bind seam). PO metadata may have arrived but the "
        "before-model hook did not write to the ContextVar.",
    ),
    (
        "SHARP REJECTION",
        re.compile(r"missing_fhir_context|sharponmcp\.com.*key-components",
                       re.IGNORECASE),
        "SHARP middleware refused the call -- X-FHIR-Server-URL or "
        "X-FHIR-Access-Token headers are absent. PO may not have generated "
        "the token for this scope; click 'Generate Token' in the dialog.",
    ),
    (
        "OAUTH FAIL",
        re.compile(r"auth_required|invalid_token|invalid_grant|"
                       r"unauthorized_client|\"code\":\s*401",
                       re.IGNORECASE),
        "Orchestrator OAuth bearer rejected. Likely expired (1h TTL) or the "
        "client-credentials path is misconfigured. Re-issue via /oauth/token.",
    ),
    (
        "AGENT EXCEPTION",
        re.compile(r"agent_error:|\"code\":\s*-32000", re.IGNORECASE),
        "A tool callable raised an unhandled exception that bubbled up to "
        "JSON-RPC. Inspect the response.error.message for the exception type.",
    ),
    (
        "DISCOVERY FALLBACK",
        re.compile(r"\"target_kind\":\s*\"discovery\"", re.IGNORECASE),
        "Engine returned the discovery catalog instead of running a "
        "workflow. Either the LLM dispatcher is unavailable / over budget "
        "or the prompt does not match any keyword route. Check the "
        "ADK_MODEL env var on the federation process.",
    ),
    (
        "WORKFLOW STEP ERROR",
        re.compile(r"\"error\":\s*\"(?!null)[^\"]+\"", re.IGNORECASE),
        "At least one workflow step has a non-null error field. Search "
        "the artifact JSON for the offending step_id; common causes are "
        "missing required inputs or downstream FHIR fetches against an "
        "absent resource.",
    ),
]


def _detect_failures(resp: dict) -> list[tuple[str, str, str]]:
    """Return [(label, hint, evidence_snippet), ...] for matching patterns."""
    blob = json.dumps(resp, ensure_ascii=False)
    hits: list[tuple[str, str, str]] = []
    for label, rx, hint in _FAIL_PATTERNS:
        m = rx.search(blob)
        if not m:
            continue
        # Snippet: 60 chars around the match for context
        s, e = m.span()
        snippet = blob[max(0, s - 30):min(len(blob), e + 30)]
        snippet = snippet.replace("\n", " ").replace("  ", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        hits.append((label, hint, snippet))
    return hits


def _render_failure_banner(hits: list[tuple[str, str, str]]) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    for label, hint, snippet in hits:
        lines.append(_bg_red(f"  !! {label} !!  "))
        lines.append(f"  {_red('hint')}:     {hint}")
        lines.append(f"  {_red('match')}:    {_grey(snippet)}")
    return "\n".join(lines) + "\n"


def _summarise_request(req: dict) -> str:
    method = req.get("method") or req.get("params", {}).get("method") or "?"
    msg = (req.get("params") or {}).get("message") or {}
    parts = msg.get("parts") or []
    text = ""
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            text = p["text"]
            break
    text = (text or "(no text)").strip().replace("\n", " ")
    if len(text) > 140:
        text = text[:137] + "..."
    meta = msg.get("metadata") or {}
    fhir = meta.get(
        "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
    ) if isinstance(meta, dict) else None
    pid = (fhir or {}).get("patientId") if isinstance(fhir, dict) else None
    return (
        f"  {_bold('method')}: {method}\n"
        f"  {_bold('prompt')}: {text}\n"
        f"  {_bold('patient')}: {_grey(pid or '(no FHIR-context metadata)')}"
    )


def _summarise_response(resp: dict, with_payload: bool = True) -> str:
    if "error" in resp:
        err = resp["error"]
        return f"  {_red('error')}: {err.get('code')} {err.get('message')}"
    result = resp.get("result") or {}
    task = result.get("task") or result
    status = (task.get("status") or {}).get("state", "?")
    artifacts = task.get("artifacts") or []
    text = ""
    msg_in_status = (task.get("status") or {}).get("message") or {}
    for p in (msg_in_status.get("parts") or []):
        if isinstance(p, dict) and p.get("text"):
            text = p["text"]
            break
    text_short = (text or "").strip().replace("\n", " // ")
    if len(text_short) > 200:
        text_short = text_short[:197] + "..."
    color = _green if "COMPLETED" in str(status).upper() else _yellow
    out = [
        f"  {_bold('state')}:    {color(str(status))}",
        f"  {_bold('artifacts')}: {len(artifacts)}",
        f"  {_bold('agent_say')}: {text_short or _grey('(empty)')}",
    ]
    if with_payload and artifacts:
        out.append(f"  {_bold('payload')}:")
        for art in artifacts:
            name = art.get("name", "?")
            desc = (art.get("description") or "").strip()
            out.append(f"    - {_blue(name)}: {desc[:120]}")
            for part in art.get("parts", []):
                if part.get("kind") == "data" or "data" in part:
                    data = part.get("data", {})
                    if isinstance(data, dict):
                        for k, v in list(data.items())[:6]:
                            v_str = json.dumps(v) if not isinstance(v, str) else v
                            if len(v_str) > 100:
                                v_str = v_str[:97] + "..."
                            out.append(f"      {_grey(k)}: {v_str}")
    return "\n".join(out)


def _render(
    entry: dict,
    with_payload: bool,
    filter_slug: str | None,
    errors_only: bool,
) -> str | None:
    req = entry.get("request") or {}
    resp = entry.get("response") or {}
    if filter_slug and filter_slug not in json.dumps(resp):
        return None
    failures = _detect_failures(resp)
    if errors_only and not failures:
        return None
    ts = entry.get("ts", "?")
    banner = _render_failure_banner(failures)
    bar = _bg_red("=" * 78) if failures else _bold("=" * 78)
    return (
        f"\n{bar}\n"
        f"{_bold(ts)}\n"
        f"{banner}"
        f"{_blue('REQUEST')}\n{_summarise_request(req)}\n"
        f"{_blue('RESPONSE')}\n{_summarise_response(resp, with_payload)}\n"
    )


def _tail_file(
    path: Path,
    replay: int,
    with_payload: bool,
    filter_slug: str | None,
    errors_only: bool,
) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with path.open("r", encoding="utf-8") as fh:
        # Replay last N lines if requested
        if replay > 0:
            lines = fh.readlines()[-replay:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rendered = _render(entry, with_payload, filter_slug, errors_only)
                if rendered:
                    print(rendered)
        else:
            fh.seek(0, 2)  # end of file

        mode = "errors-only" if errors_only else "all-traces"
        print(_grey(
            f"[watching {path}]  ({path.stat().st_size} bytes; mode={mode}; "
            f"Ctrl-C to stop)"
        ))
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.4)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rendered = _render(entry, with_payload, filter_slug, errors_only)
            if rendered:
                print(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=0,
                          help="Replay the last N entries before tailing.")
    parser.add_argument("--no-payload", action="store_true",
                          help="Drop the artifact data block from each render.")
    parser.add_argument("--filter", type=str, default=None,
                          help="Only print entries containing this slug.")
    parser.add_argument("--errors-only", action="store_true",
                          help="Only print entries that match a fail pattern.")
    args = parser.parse_args()
    try:
        _tail_file(
            TRACE, args.tail, not args.no_payload, args.filter,
            args.errors_only,
        )
    except KeyboardInterrupt:
        print()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
