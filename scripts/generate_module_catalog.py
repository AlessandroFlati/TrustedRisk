"""Phase 17.CON2 - Auto-discover + render the a2a_agent module
catalog.

Walks ``src/a2a_agent/*.py``, extracts:
  - module name + first-line docstring summary
  - public Pydantic models (subclasses of BaseModel)
  - public free functions (non-underscore)
  - module size (LoC) for ranking

Emits ``docs/MODULE_CATALOG.md`` + ``docs/module_catalog.json``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel


def _extract_first_line_summary(docstring: str | None) -> str:
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _module_loc(path: Path) -> int:
    try:
        return sum(
            1 for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
    except OSError:
        return 0


def _walk_a2a_agent() -> list[dict[str, Any]]:
    pkg_dir = ROOT / "src" / "a2a_agent"
    catalog: list[dict[str, Any]] = []
    for py_path in sorted(pkg_dir.glob("*.py")):
        if py_path.name in ("__init__.py", "_internal.py"):
            continue
        mod_name = f"a2a_agent.{py_path.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            catalog.append({
                "module": mod_name,
                "import_error": f"{type(exc).__name__}: {exc}",
                "loc": _module_loc(py_path),
                "summary": "",
                "models": [],
                "functions": [],
            })
            continue
        summary = _extract_first_line_summary(mod.__doc__)
        models: list[str] = []
        functions: list[str] = []
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and issubclass(obj, BaseModel) \
                    and obj.__module__ == mod_name:
                models.append(name)
            elif (inspect.isfunction(obj)
                  or inspect.iscoroutinefunction(obj)) \
                    and obj.__module__ == mod_name:
                functions.append(name)
        catalog.append({
            "module": mod_name,
            "loc": _module_loc(py_path),
            "summary": summary,
            "models": sorted(models),
            "functions": sorted(functions),
        })
    return catalog


def _render_md(entries: list[dict[str, Any]]) -> str:
    n_modules = len(entries)
    n_with_models = sum(1 for e in entries if e.get("models"))
    n_total_models = sum(
        len(e.get("models", [])) for e in entries
    )
    n_total_functions = sum(
        len(e.get("functions", [])) for e in entries
    )
    total_loc = sum(e.get("loc", 0) for e in entries)
    lines = [
        "# TrustedRisk - a2a_agent Module Catalog",
        "",
        f"**Modules**: {n_modules} - "
        f"**Modules with Pydantic models**: {n_with_models} - "
        f"**Total Pydantic models**: {n_total_models} - "
        f"**Total public functions**: {n_total_functions} - "
        f"**Total LoC (non-blank, non-comment)**: {total_loc:,}",
        "",
    ]
    for e in sorted(entries, key=lambda e: e["module"]):
        if e.get("import_error"):
            lines.append(f"## `{e['module']}` _(import error)_")
            lines.append(f"- error: `{e['import_error']}`")
            lines.append("")
            continue
        lines.append(f"## `{e['module']}`")
        if e["summary"]:
            lines.append(f"_{e['summary']}_")
        lines.append("")
        lines.append(f"- **LoC**: {e['loc']}")
        if e["models"]:
            lines.append(
                f"- **Pydantic models** ({len(e['models'])}): "
                + ", ".join(f"`{m}`" for m in e["models"])
            )
        if e["functions"]:
            lines.append(
                f"- **Public functions** ({len(e['functions'])}): "
                + ", ".join(f"`{f}`" for f in e["functions"])
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    out_dir = ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _walk_a2a_agent()
    md = _render_md(entries)
    (out_dir / "MODULE_CATALOG.md").write_text(md, encoding="utf-8")
    (out_dir / "module_catalog.json").write_text(
        json.dumps(entries, indent=2),
        encoding="utf-8",
    )
    n_total_models = sum(len(e.get("models", [])) for e in entries)
    n_total_functions = sum(len(e.get("functions", [])) for e in entries)
    print(
        f"Catalog: {len(entries)} modules, {n_total_models} models, "
        f"{n_total_functions} functions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
