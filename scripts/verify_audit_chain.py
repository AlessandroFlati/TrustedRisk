"""Phase 12.2 -- Audit Merkle proof verifier CLI.

Reads a JSON file describing:
    {
      "events": [<event dict>, ...],
      "target_event_id": "...",
      "claimed_leaf_hash": "<sha256 hex>",
      "proof_steps": [{"sibling_hash": "...", "is_left": true|false}, ...],
      "expected_merkle_root": "<sha256 hex>"
    }

Prints the structured `MerkleAuditVerificationReport`. Exit code:
  0 -- verified
  1 -- tamper_event / tamper_chain / unknown_event
  2 -- bad input

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_audit_chain.py \\
        path/to/proof.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.merkle_audit import verify_audit_chain


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a TrustedRisk Merkle audit-chain inclusion proof.",
    )
    parser.add_argument("proof_path", help="Path to the proof JSON file.")
    args = parser.parse_args()

    fp = Path(args.proof_path)
    if not fp.exists():
        print(f"ERROR: proof file not found at {fp}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON: {exc}", file=sys.stderr)
        return 2

    required = ("events", "target_event_id", "claimed_leaf_hash",
                    "proof_steps", "expected_merkle_root")
    missing = [k for k in required if k not in payload]
    if missing:
        print(f"ERROR: missing keys in proof file: {missing}",
                  file=sys.stderr)
        return 2

    report = verify_audit_chain(
        events=payload["events"],
        target_event_id=payload["target_event_id"],
        claimed_leaf_hash=payload["claimed_leaf_hash"],
        proof_steps=payload["proof_steps"],
        expected_merkle_root=payload["expected_merkle_root"],
    )

    print(json.dumps(report.model_dump(), indent=2, default=str))
    return 0 if report.posture == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
