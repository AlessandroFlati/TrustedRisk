"""Multi-prompt A2A eval harness.

Runs the curated query suite (eval_queries.json) through TrustedRisk's
refusal classifier (always) and optionally through a live A2A agent if
ADK_MODEL is set + LLM creds are available.

Usage:
    # Stub-only (no LLM, just classifier)
    python tests/eval/run_eval.py

    # With live A2A (requires Gemini key OR Ollama running, ADK_MODEL set)
    python tests/eval/run_eval.py --live

Outputs:
    tests/eval/eval_report.md   -- markdown summary
    tests/eval/eval_results.json -- per-query raw results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from a2a_agent.refusal_classifier import classify_query  # noqa: E402


def load_queries() -> dict:
    fp = Path(__file__).parent / "eval_queries.json"
    with fp.open(encoding="utf-8") as f:
        return json.load(f)


def run_classifier(queries: list[dict]) -> list[dict]:
    """Run each query through the deterministic classifier; record outcome."""
    out = []
    for q in queries:
        t0 = time.perf_counter()
        result = classify_query(q["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = result.category == q["expected_outcome"]
        out.append({
            "id": q["id"],
            "text": q["text"],
            "expected": q["expected_outcome"],
            "actual": result.category,
            "matched_pattern": result.matched_pattern,
            "rationale": result.rationale,
            "passed": ok,
            "elapsed_ms": round(elapsed_ms, 3),
        })
    return out


async def run_live_agent(queries: list[dict]) -> list[dict]:
    """Run queries through the actual A2A root agent (LLM-backed).

    Requires ADK_MODEL to be set + the corresponding API key.
    """
    try:
        from a2a_agent.agent import root_agent
    except RuntimeError as e:
        print(f"Live agent unavailable: {e}", file=sys.stderr)
        return []

    agent = root_agent()
    out = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            response = await agent.run_async(q["text"])  # type: ignore[attr-defined]
            text = str(response)[:500]
            err = None
        except Exception as e:
            text = ""
            err = f"{type(e).__name__}: {e}"
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out.append({
            "id": q["id"],
            "text": q["text"],
            "expected": q["expected_outcome"],
            "agent_response_excerpt": text,
            "error": err,
            "elapsed_ms": round(elapsed_ms, 3),
        })
    return out


def emit_markdown(suite: dict, classifier_results: list[dict],
                  live_results: list[dict] | None) -> str:
    lines: list[str] = []
    lines.append("# TrustedRisk multi-prompt eval report")
    lines.append("")
    lines.append(
        f"**Suite:** {suite.get('description', '')[:200]}\n"
    )
    lines.append(f"**Queries:** {len(classifier_results)}")
    lines.append("")

    # Aggregate
    pass_count = sum(1 for r in classifier_results if r["passed"])
    total = len(classifier_results)
    pass_rate = pass_count / total if total else 0
    lines.append(f"**Refusal-classifier pass rate:** {pass_count}/{total} ({pass_rate:.1%})")
    lines.append("")

    # By category
    by_cat: dict[str, list[dict]] = {}
    for r in classifier_results:
        by_cat.setdefault(r["expected"], []).append(r)
    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| Category | Total | Passed | Rate |")
    lines.append("|---|---:|---:|---:|")
    for cat, items in sorted(by_cat.items()):
        ok = sum(1 for x in items if x["passed"])
        lines.append(f"| `{cat}` | {len(items)} | {ok} | {ok/len(items):.0%} |")
    lines.append("")

    # Misclassifications
    misses = [r for r in classifier_results if not r["passed"]]
    if misses:
        lines.append("## Misclassifications")
        lines.append("")
        for r in misses:
            lines.append(f"- **{r['id']}** -- expected `{r['expected']}`, got `{r['actual']}`")
            lines.append(f"  - Query: {r['text']!r}")
            lines.append(f"  - Rationale: {r['rationale']}")
        lines.append("")
    else:
        lines.append("No misclassifications.\n")

    # All results table
    lines.append("## Per-query results")
    lines.append("")
    lines.append("| ID | Expected | Actual | Pass | Pattern | Latency |")
    lines.append("|---|---|---|:---:|---|---:|")
    for r in classifier_results:
        check = "✓" if r["passed"] else "✗"
        pat = (r["matched_pattern"] or "--")[:40]
        lines.append(
            f"| {r['id']} | `{r['expected']}` | `{r['actual']}` | {check} | "
            f"`{pat}` | {r['elapsed_ms']:.2f}ms |"
        )
    lines.append("")

    # Live agent section
    if live_results:
        lines.append("## Live A2A agent results")
        lines.append("")
        lines.append(
            f"Model: `{os.environ.get('ADK_MODEL', 'unknown')}`. "
            f"Note: live results are LLM-dependent -- run multiple times to assess "
            f"variance, especially on ambiguous boundary cases."
        )
        lines.append("")
        lines.append("| ID | Expected | Latency | Excerpt |")
        lines.append("|---|---|---:|---|")
        for r in live_results:
            excerpt = (r["agent_response_excerpt"] or r.get("error") or "")[:80]
            lines.append(f"| {r['id']} | `{r['expected']}` | {r['elapsed_ms']:.0f}ms | {excerpt!r} |")
        lines.append("")

    # Methodology
    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "1. **Classifier** is the deterministic pre-LLM gate "
        "(`src/a2a_agent/refusal_classifier.py`) that mirrors the rule set "
        "documented in `src/a2a_agent/instructions.md`. Its purpose is to "
        "short-circuit obvious refuse-cases before paying for LLM round-trip "
        "and to provide a regression-testable surface."
    )
    lines.append("")
    lines.append(
        "2. **Live A2A** (when run with `--live`) exercises the full root agent "
        "-> sub-agent routing -> MCP tool invocation chain. It validates that the "
        "LLM agrees with the deterministic classifier on routing decisions and "
        "produces well-formed Decision Cards / refusal messages."
    )
    lines.append("")
    lines.append(
        "3. The eval suite is **not** a clinical accuracy benchmark -- it tests "
        "the **safety policy** and **routing robustness**. Clinical accuracy is "
        "validated separately by the W1 calibration ECE gate (preferred ≤0.05)."
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Also invoke the live A2A agent (requires LLM creds)")
    args = parser.parse_args()

    suite = load_queries()
    queries = suite["queries"]

    classifier_results = run_classifier(queries)

    live_results: list[dict] | None = None
    if args.live:
        live_results = asyncio.run(run_live_agent(queries))

    md = emit_markdown(suite, classifier_results, live_results)
    out_md = Path(__file__).parent / "eval_report.md"
    out_md.write_text(md, encoding="utf-8")

    out_json = Path(__file__).parent / "eval_results.json"
    out_json.write_text(json.dumps({
        "suite_meta": {k: v for k, v in suite.items() if k != "queries"},
        "classifier": classifier_results,
        "live": live_results,
    }, indent=2), encoding="utf-8")

    pass_count = sum(1 for r in classifier_results if r["passed"])
    total = len(classifier_results)
    print(f"Classifier: {pass_count}/{total} passed ({pass_count/total:.0%})")
    print(f"Report:  {out_md}")
    print(f"Results: {out_json}")
    return 0 if pass_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
