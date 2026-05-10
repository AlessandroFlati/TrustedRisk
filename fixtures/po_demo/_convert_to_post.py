"""Rewrite the demo bundles from PUT-with-id (updateCreate) to POST-with-urn:uuid.

Some FHIR servers (Prompt Opinion's workspace store, for instance) do not
expose the `updateCreate` capability, so a transaction Bundle that PUTs to
`Patient/demo-eleanor-greene` rejects with `not-supported`. This script
rewrites the four bundles in place to use POST + the canonical
`fullUrl: urn:uuid:<UUID>` shape that any FHIR R4 server accepts:

  - every entry's `fullUrl` becomes `urn:uuid:<RFC-4122-UUID>` (deterministic
    UUIDv5 derived from the original slug + a fixed namespace, so re-running
    the script is idempotent and the same slug always maps to the same UUID)
  - every entry's `request` becomes `{method: "POST", url: "<resourceType>"}`
  - every internal `reference` of the form `<ResourceType>/<id>` is
    rewritten to point to the corresponding `urn:uuid:<UUID>` so the server
    can resolve them inside the transaction.

The PO workspace FHIR server validates `fullUrl` against RFC 4122, so plain
slugs like `urn:uuid:demo-eleanor-greene` are rejected with
"is not a correct literal for an uri".

Run from the repo root:
    .venv/Scripts/python.exe fixtures/po_demo/_convert_to_post.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
BUNDLES = sorted(DEMO_DIR.glob("*.json"))

# Stable namespace so the same slug always maps to the same UUID across
# every machine that re-runs this script. The exact value is arbitrary --
# what matters is that it never changes.
_NS = uuid.UUID("8e2f1c0a-3b4d-5e6f-7a8b-9c0d1e2f3a4b")


def _slug_to_urn(slug: str) -> str:
    return f"urn:uuid:{uuid.uuid5(_NS, slug)}"


def _walk(node, ref_to_uuid: dict[str, str]):
    """Recursively rewrite reference strings inside a JSON value."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "reference" and isinstance(v, str) and v in ref_to_uuid:
                node[k] = ref_to_uuid[v]
            else:
                _walk(v, ref_to_uuid)
    elif isinstance(node, list):
        for item in node:
            _walk(item, ref_to_uuid)


def convert(path: Path) -> None:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("type") != "transaction":
        return

    # The bundle may already have urn:uuid: fullUrls (from a prior run of
    # this script with a buggy slug-as-uuid format). Rebuild the slug map
    # from BOTH the resource.id (PUT-shaped source) and the fullUrl
    # (already-converted source) so the script is idempotent.
    ref_to_uuid: dict[str, str] = {}

    def _slug_of(entry: dict) -> str | None:
        res = entry.get("resource") or {}
        rid = res.get("id")
        if rid:
            return rid
        full = entry.get("fullUrl") or ""
        if full.startswith("urn:uuid:"):
            return full[len("urn:uuid:"):]
        return None

    # Pass 1: build map slug -> urn:uuid:<UUID>
    slug_to_urn: dict[str, str] = {}
    for entry in bundle.get("entry") or []:
        slug = _slug_of(entry)
        if slug:
            slug_to_urn[slug] = _slug_to_urn(slug)

    # Build the reference-rewrite map for two shapes:
    #   "Patient/demo-eleanor-greene"  ->  urn:uuid:<UUID>
    #   "urn:uuid:demo-eleanor-greene" ->  urn:uuid:<UUID>   (re-conversion)
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") or {}
        rtype = res.get("resourceType")
        slug = _slug_of(entry)
        if not slug:
            continue
        new_urn = slug_to_urn[slug]
        if rtype:
            ref_to_uuid[f"{rtype}/{slug}"] = new_urn
        ref_to_uuid[f"urn:uuid:{slug}"] = new_urn

    # Pass 2: rewrite each entry.
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") or {}
        slug = _slug_of(entry)
        rtype = res.get("resourceType")
        if slug and rtype:
            entry["fullUrl"] = slug_to_urn[slug]
            entry["request"] = {"method": "POST", "url": rtype}
            # Strip the client-assigned id; the server will assign one.
            res.pop("id", None)
        _walk(res, ref_to_uuid)

    path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"converted: {path.name}")


for path in BUNDLES:
    if path.name.startswith("_"):
        continue
    convert(path)
