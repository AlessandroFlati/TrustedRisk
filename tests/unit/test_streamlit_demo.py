"""Phase 12.10 A1 -- Streamlit browser demo smoke tests.

The full Streamlit runtime is not available in CI, but we can still
prove the app file is well-formed Python that imports cleanly when
streamlit is stubbed out.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
APP_PATH = REPO_ROOT / "apps" / "streamlit_demo" / "app.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ─────────────────────────────────────────────────────────────────────
# Static smoke
# ─────────────────────────────────────────────────────────────────────

def test_app_file_parses_as_valid_python():
    text = APP_PATH.read_text(encoding="utf-8")
    ast.parse(text)  # raises SyntaxError on malformed source


def test_app_file_declares_at_least_8_tabs():
    """The demo UI ships 8 tabs -- Scenarios, Counterfactuals, RT v2,
    RT v3, Calibration, Performance, MedQA, Tool catalogue."""
    text = APP_PATH.read_text(encoding="utf-8")
    for tab_label in (
        "Scenarios", "Counterfactuals",
        "Red-team v2", "Red-team v3 map",
        "Calibration", "Performance",
        "MedQA bench", "Tool catalogue",
    ):
        assert f'"{tab_label}"' in text, (
            f"Demo UI is missing tab {tab_label!r}"
        )


def test_streamlit_listed_as_optional_demo_dependency():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "streamlit" in text


def test_makefile_exposes_streamlit_demo_target():
    mk = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "streamlit-demo:" in mk
    assert "streamlit run" in mk


# ─────────────────────────────────────────────────────────────────────
# Stubbed-streamlit import smoke
# ─────────────────────────────────────────────────────────────────────

def test_app_imports_with_streamlit_stub(monkeypatch, tmp_path):
    """Replace `streamlit` with a no-op stub, run the app file
    top-level, and assert it doesn't raise. This catches signature
    drift in the federation modules the demo imports (scenarios,
    counterfactuals, planner, etc.)."""
    if "streamlit" in sys.modules:
        # Real streamlit installed -- skip the stub path.
        pytest.skip("streamlit installed; stub-import path skipped")

    fake = types.ModuleType("streamlit")

    class _NullCtx:
        def __enter__(self_inner): return self_inner
        def __exit__(self_inner, *a): return False
        def __getattr__(self_inner, name):
            return _Recorder()

    class _Recorder:
        def __call__(self, *args, **kwargs):
            return _NullCtx()
        def __enter__(self_inner): return self_inner
        def __exit__(self_inner, *a): return False
        def __getattr__(self_inner, name):
            return _Recorder()

    def _no_op(*args, **kwargs):
        return _NullCtx()

    # Wire the public Streamlit surface the app uses
    fake.set_page_config = _no_op
    fake.title = _no_op
    fake.caption = _no_op
    fake.tabs = lambda labels: [_NullCtx() for _ in labels]
    fake.subheader = _no_op
    fake.write = _no_op
    fake.warning = _no_op
    fake.success = _no_op
    fake.metric = _no_op
    fake.dataframe = _no_op
    fake.json = _no_op
    fake.selectbox = lambda label, options, **k: options[0] if options else None
    fake.button = lambda label, **k: False  # never click
    fake.expander = lambda title, **k: _NullCtx()
    fake.columns = lambda n, **k: [_NullCtx() for _ in range(n)]
    monkeypatch.setitem(sys.modules, "streamlit", fake)

    # Execute the file in a fresh namespace
    src = APP_PATH.read_text(encoding="utf-8")
    namespace: dict = {"__name__": "apps.streamlit_demo.app",
                            "__file__": str(APP_PATH)}
    exec(compile(src, str(APP_PATH), "exec"), namespace, namespace)
