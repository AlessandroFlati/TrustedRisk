"""Unified A2A federation server -- all 16 agents in one process.

Topology (vs the per-process design in `scripts/demo_up_full.sh`):

    Caddy (:8181)
        |
        | /a2a/*  ->  127.0.0.1:9001  (this app)
        v
    Starlette outer
        |
        +-- Mount /a2a/trustedrisk-discharge   -> build_specialist_app(discharge_card)
        +-- Mount /a2a/trustedrisk-acute       -> build_specialist_app(acute_card)
        +-- ... 13 more specialists ...
        +-- Mount /a2a/trustedrisk-composer    -> apps.composer.server:app
        +-- Route  /healthz                    -> aggregated liveness

Each mounted sub-app keeps its own SHARP / OAuth / rate-limit middleware
stack; the outer wrapper does not introduce extra middleware. FastMCP
tool registrations happen once per sub-app at import time, but the
underlying `mcp_server.tools.BUNDLES` is module-level so the heavy
import work is shared.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe -m apps.federation.server
or override port:
    TRUSTEDRISK_FEDERATION_PORT=9001 ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from apps._shared import SpecialistConfig, build_specialist_app


class _StripRootPath:
    """Rebuild `scope.path` so it no longer contains the mount prefix.

    Starlette 1.0+ keeps the original path in `scope.path` after Mount
    dispatch and updates only `scope.root_path`. The MCP sub-apps key
    their SHARP / OAuth whitelists off `scope.path`, so without this
    wrapper a request to `/a2a/<name>/.well-known/agent-card.json`
    arrives at the sub-app as `path == "/a2a/<name>/.well-known/..."`
    and fails the public-prefix match.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            root = scope.get("root_path", "") or ""
            path = scope.get("path", "") or ""
            if root and path.startswith(root):
                scope = dict(scope)
                scope["path"] = path[len(root):] or "/"
        await self.app(scope, receive, send)


class _AddTrailingSlashForMounts:
    """Rewrite scope.path to add a trailing slash on bare Mount prefix
    requests, preventing Starlette's automatic 307 redirect.

    A2A v1 chat clients (Prompt Opinion, etc.) POST to the URL declared
    in the agent card -- a bare `/a2a/<name>` with no trailing slash.
    Starlette's Mount routing answers a 307 to canonicalise to
    `/a2a/<name>/`. HTTP 307 instructs clients to drop the Authorization
    header (security rule against credential leakage on cross-path
    redirects), so the retried request lands unauthenticated and OAuth
    rejects with 401. Rewriting in-process keeps the original request
    intact, headers and all.
    """

    def __init__(self, app, prefixes):
        self.app = app
        self.prefixes = tuple(prefixes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            path = scope.get("path", "") or ""
            if path in self.prefixes:
                scope = dict(scope)
                scope["path"] = path + "/"
        await self.app(scope, receive, send)


def _specialist_card_paths() -> list[Path]:
    return sorted((ROOT / "apps").glob("specialist_*/agent_card.json"))


def _agent_url_slug(card: dict) -> str:
    """Extract the URL-stable slug for the mount path from the card.

    The display `name` field is now a human-readable label (e.g.
    "Discharge Planner"), so we derive the routing slug from the
    declared URL instead -- last path segment of `url`.
    """
    url = card.get("url") or ""
    return url.rstrip("/").rsplit("/", 1)[-1] or card.get("_specialist", "agent")


def _mount_for_specialist(card_path: Path) -> Mount:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    cfg = SpecialistConfig(
        name=card.get("name", "specialist"),
        port=0,
        agent_card_path=str(card_path),
    )
    sub_app = build_specialist_app(cfg)
    slug = _agent_url_slug(card)
    return Mount(f"/a2a/{slug}", app=_StripRootPath(sub_app))


def _mount_for_composer() -> Mount:
    from apps.composer.server import app as composer_app
    return Mount("/a2a/trustedrisk-composer", app=_StripRootPath(composer_app))


def _mount_for_orchestrator() -> Mount:
    from apps.orchestrator.server import app as orchestrator_app
    return Mount("/a2a/trustedrisk-orchestrator",
                  app=_StripRootPath(orchestrator_app))


async def healthz(_request):
    cards = _specialist_card_paths()
    composer_card_path = ROOT / "apps" / "composer" / "agent_card.json"
    orchestrator_card_path = ROOT / "apps" / "orchestrator" / "agent_card.json"
    names: list[str] = []
    for p in [*cards, composer_card_path, orchestrator_card_path]:
        try:
            names.append(json.loads(p.read_text(encoding="utf-8"))["name"])
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return JSONResponse({
        "status": "ok",
        "n_agents": len(names),
        "agents": sorted(names),
    })


async def well_known_marketplace(_request):
    """Aggregate marketplace manifest covering every mounted agent."""
    manifest_path = ROOT / "docs" / "federation" / "marketplace_manifest.json"
    if not manifest_path.exists():
        return JSONResponse(
            status_code=503,
            content={"error": "marketplace_manifest_missing"},
        )
    return JSONResponse(json.loads(manifest_path.read_text(encoding="utf-8")))


def build_federation_app():
    mount_prefixes: list[str] = []
    routes: list = []
    for p in _specialist_card_paths():
        m = _mount_for_specialist(p)
        mount_prefixes.append(m.path)
        routes.append(m)
    composer_mount = _mount_for_composer()
    mount_prefixes.append(composer_mount.path)
    routes.append(composer_mount)
    orchestrator_mount = _mount_for_orchestrator()
    mount_prefixes.append(orchestrator_mount.path)
    routes.append(orchestrator_mount)
    routes.append(Route("/healthz", healthz, methods=["GET"]))
    routes.append(Route(
        "/.well-known/marketplace.json",
        well_known_marketplace,
        methods=["GET"],
    ))
    inner = Starlette(routes=routes)
    return _AddTrailingSlashForMounts(inner, mount_prefixes)


app = build_federation_app()


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_FEDERATION_HOST", "127.0.0.1")
    port = int(os.environ.get("TRUSTEDRISK_FEDERATION_PORT", "9001"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
