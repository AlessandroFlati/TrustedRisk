"""In-memory FHIR context helpers for offline tests.

Provides a context manager that monkey-patches `fetch_patient_bundle`
and `resolve_patient_id` in `mcp_server.fhir.client` so tests can run
workflows and dispatcher routes without an HTTP connection to a FHIR
server.

Usage::

    from tests.fixtures.fhir_helpers import bind_in_memory_fhir

    MY_BUNDLE = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "test-01",
                          "birthDate": "1955-01-01", "gender": "female"}},
        ],
    }

    async def test_something():
        with bind_in_memory_fhir(MY_BUNDLE, patient_id="Patient/test-01"):
            result = await execute_workflow(wf, inputs)
            assert result.abstain_recommended is False
"""

from __future__ import annotations

import json
import pathlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------- helpers


def load_fixture(name: str) -> dict:
    """Load a FHIR bundle JSON from ``tests/fixtures/fhir/<name>.json``.

    Args:
        name: Filename stem (without ``.json``), e.g. ``"minimal_chf_patient"``.

    Returns:
        The parsed bundle dict.

    Raises:
        FileNotFoundError: if the fixture file does not exist.
        ValueError: if the file content is not a valid JSON object.
    """
    fixture_path = pathlib.Path(__file__).parent / "fhir" / f"{name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"FHIR fixture not found: {fixture_path}. "
            f"Create it under tests/fixtures/fhir/{name}.json"
        )
    with open(fixture_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Fixture {name!r}: expected a JSON object (dict), got "
            f"{type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------- context manager


@contextmanager
def bind_in_memory_fhir(
    bundle: dict,
    *,
    patient_id: str = "Patient/test",
):
    """Bind an in-memory FHIR bundle for the duration of the ``with`` block.

    Monkey-patches ``mcp_server.fhir.client.fetch_patient_bundle`` and
    ``mcp_server.fhir.client.resolve_patient_id`` so the dispatcher and
    any tool that calls those functions receives the supplied bundle
    without making an HTTP request.

    Both functions are patched at the module level
    (``mcp_server.fhir.client``) so imports that do
    ``from mcp_server.fhir.client import fetch_patient_bundle`` inside a
    function body (as the dispatcher does) also pick up the patch.

    Args:
        bundle: A FHIR R4 Bundle dict to return from
            ``fetch_patient_bundle``.
        patient_id: The ``patient_id`` value that
            ``resolve_patient_id`` will return. Should match the
            ``Patient.id`` in the bundle (prefixed with ``"Patient/"``).

    Yields:
        None -- use the ``with`` block to run the code under test.

    Example::

        with bind_in_memory_fhir(MY_BUNDLE, patient_id="Patient/pt-test"):
            out = await execute_workflow(wf, inputs)
    """
    if not isinstance(bundle, dict):
        raise TypeError(
            f"bundle must be a dict, got {type(bundle).__name__}"
        )

    async def _mock_fetch(pid: str) -> dict[str, Any]:
        return bundle

    async def _mock_resolve(explicit: str | None = None) -> str:
        return explicit or patient_id

    with (
        patch(
            "mcp_server.fhir.client.fetch_patient_bundle",
            side_effect=_mock_fetch,
        ),
        patch(
            "mcp_server.fhir.client.resolve_patient_id",
            side_effect=_mock_resolve,
        ),
    ):
        yield
