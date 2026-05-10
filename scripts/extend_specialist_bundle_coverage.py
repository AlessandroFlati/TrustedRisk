"""Phase 17.AI - Extend specialist agent_card.json to cover Phase 13/14
bundles.

Given the bundle-to-owner map in ``federation_registry``, walks every
existing ``apps/specialist_*/agent_card.json`` and adds any missing
bundles assigned to that specialist (with the canonical tool list
from ``mcp_server.tools.BUNDLES``).

Idempotent: re-running on an already-extended card is a no-op.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.federation_registry import (
    _BUNDLE_TO_OWNER, build_federation_registry,
)


def main() -> int:
    apps_dir = ROOT / "apps"
    reg = build_federation_registry(apps_dir=apps_dir)
    if not reg.bundles_missing:
        print("All runtime bundles are already covered.")
        return 0
    from mcp_server.tools import BUNDLES   # type: ignore

    # Index existing cards by canonical slug (`_legacy_name`).
    # The human-readable `name` is reserved for the chat-client dropdown.
    cards_by_name: dict[str, Path] = {}
    for path in sorted(apps_dir.glob("specialist_*/agent_card.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        slug = data.get("_legacy_name") or data.get("name")
        if slug:
            cards_by_name[slug] = path

    n_extended = 0
    for missing_bundle in reg.bundles_missing:
        owner = _BUNDLE_TO_OWNER.get(missing_bundle)
        if not owner or owner not in cards_by_name:
            print(f"  skip `{missing_bundle}` -> owner `{owner}` "
                  "missing card")
            continue
        card_path = cards_by_name[owner]
        data = json.loads(card_path.read_text(encoding="utf-8"))
        bundles_dict = data.setdefault("_bundles", {})
        if missing_bundle in bundles_dict:
            continue
        bundles_dict[missing_bundle] = list(BUNDLES[missing_bundle])
        card_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        n_extended += 1
        print(f"  added `{missing_bundle}` -> `{owner}`")

    print(f"Extended {n_extended} bundle-owner pairings.")
    reg2 = build_federation_registry(apps_dir=apps_dir)
    print(
        f"Coverage now: {len(reg2.bundles_covered)}/"
        f"{len(reg2.bundles_covered) + len(reg2.bundles_missing)} "
        f"({reg2.coverage_percent:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
