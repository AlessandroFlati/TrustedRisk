"""Phase 3.4 -- A2A Push notification config registry + dispatcher.

Per the A2A v1 spec, an agent that declares
`capabilities.pushNotifications = true` accepts:

  - `CreateTaskPushNotificationConfig` -> register a webhook URL for a
    given taskId. The server then POSTs `TaskStatusUpdateEvent` /
    `TaskArtifactUpdateEvent` payloads to the webhook as the task
    progresses.
  - `DeleteTaskPushNotificationConfig` -> remove the registration.

This module provides the in-memory config registry + the async
dispatcher. For this prototype we use an in-memory dict; production
deployments would back this with Redis or a database so multiple
worker processes share the registry.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class PushNotificationConfig:
    """A single push-notification registration."""
    config_id: str
    task_id: str
    webhook_url: str
    auth_header: str | None = None
    severity_threshold: str | None = None
    created_at: float = field(default_factory=time.time)


class PushNotificationRegistry:
    """In-process registry of push-notification configs.

    Keyed by `config_id`. Lookup-by-task is O(N) -- fine for the demo;
    a production deployment would index by `task_id` and persist to
    Redis so multiple workers see the same registry.
    """

    def __init__(self) -> None:
        self._configs: dict[str, PushNotificationConfig] = {}

    def create(
        self,
        task_id: str,
        webhook_url: str,
        *,
        auth_header: str | None = None,
        severity_threshold: str | None = None,
        config_id: str | None = None,
    ) -> PushNotificationConfig:
        cfg = PushNotificationConfig(
            config_id=config_id or f"push-{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            webhook_url=webhook_url,
            auth_header=auth_header,
            severity_threshold=severity_threshold,
        )
        self._configs[cfg.config_id] = cfg
        return cfg

    def delete(self, config_id: str) -> bool:
        return self._configs.pop(config_id, None) is not None

    def list_for_task(self, task_id: str) -> list[PushNotificationConfig]:
        return [c for c in self._configs.values() if c.task_id == task_id]

    def list_all(self) -> list[PushNotificationConfig]:
        return list(self._configs.values())

    def clear(self) -> None:
        self._configs.clear()


# Module-level default registry -- production deployments may swap this
# for a Redis-backed implementation by re-binding `_default`.
_default = PushNotificationRegistry()


def default_registry() -> PushNotificationRegistry:
    return _default


# ─────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────

async def dispatch_task_status_update(
    task_id: str,
    *,
    state: str,
    context_id: str | None = None,
    progress_pct: float | None = None,
    message: str | None = None,
    severity: str | None = None,
    registry: PushNotificationRegistry | None = None,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    """POST a TaskStatusUpdateEvent to every registered webhook for the task.

    Returns the list of (config_id, status_code, error?) reports -- used
    by tests to assert dispatch behaviour. Non-2xx responses or network
    errors are reported but do NOT raise; the dispatcher is best-
    effort (consistent with the A2A spec's at-most-once delivery).
    """
    registry = registry or _default
    configs = registry.list_for_task(task_id)
    payload: dict[str, Any] = {
        "kind": "TaskStatusUpdateEvent",
        "taskId": task_id,
        "status": {
            "state": state,
            "timestamp": int(time.time() * 1000),
        },
    }
    if context_id is not None:
        payload["contextId"] = context_id
    if progress_pct is not None:
        payload["status"]["progress_pct"] = round(progress_pct, 4)
    if message is not None:
        payload["status"]["message"] = {
            "role": "ROLE_AGENT",
            "parts": [{"kind": "text", "text": message}],
        }

    return await _dispatch(payload, configs, severity=severity, timeout_s=timeout_s)


async def dispatch_task_artifact_update(
    task_id: str,
    artifact_payload: Any,
    *,
    artifact_id: str | None = None,
    context_id: str | None = None,
    severity: str | None = None,
    registry: PushNotificationRegistry | None = None,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    registry = registry or _default
    configs = registry.list_for_task(task_id)
    payload: dict[str, Any] = {
        "kind": "TaskArtifactUpdateEvent",
        "taskId": task_id,
        "artifact": {
            "artifactId": artifact_id or f"artifact-{uuid.uuid4().hex[:8]}",
            "parts": [{"kind": "data", "data": artifact_payload}],
        },
    }
    if context_id is not None:
        payload["contextId"] = context_id
    return await _dispatch(payload, configs, severity=severity, timeout_s=timeout_s)


_SEVERITY_RANK = {"info": 0, "low": 1, "warn": 2, "alert": 3, "critical": 4}


def _matches_threshold(threshold: str | None, severity: str | None) -> bool:
    """True if `severity` is at least as high as `threshold`. Missing
    threshold = always match. Missing severity = always match."""
    if not threshold or not severity:
        return True
    t = _SEVERITY_RANK.get(threshold.lower(), 0)
    s = _SEVERITY_RANK.get(severity.lower(), 0)
    return s >= t


async def _dispatch(
    payload: dict[str, Any],
    configs: list[PushNotificationConfig],
    *,
    severity: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    if not configs:
        return []

    async def _post(cfg: PushNotificationConfig) -> dict[str, Any]:
        if not _matches_threshold(cfg.severity_threshold, severity):
            return {
                "config_id": cfg.config_id,
                "skipped": True,
                "reason": "below_severity_threshold",
            }
        headers = {"Content-Type": "application/json"}
        if cfg.auth_header:
            headers["Authorization"] = cfg.auth_header
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    cfg.webhook_url, json=payload, headers=headers,
                )
            return {
                "config_id": cfg.config_id,
                "status_code": resp.status_code,
                "ok": 200 <= resp.status_code < 300,
            }
        except httpx.HTTPError as exc:
            return {
                "config_id": cfg.config_id,
                "status_code": None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return await asyncio.gather(*[_post(c) for c in configs])
