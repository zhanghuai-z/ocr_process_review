"""Architecture import ratchet.

This is intentionally a ratchet, not a hard ban.  The baseline records the
current known package-level dependency violations and the test fails only when
new violating edges are introduced.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


APP_DIR = Path("app")
BASELINE_PATH = Path("architecture_baseline.json")

RULES = {
    "models_to_core": ("app.models", "app.core"),
    "core_to_engines": ("app.core", "app.engines"),
    "engines_to_core": ("app.engines", "app.core"),
    "ui_to_engines": ("app.ui", "app.engines"),
}


def test_no_new_package_import_violations():
    baseline = {
        name: set(items)
        for name, items in json.loads(BASELINE_PATH.read_text(encoding="utf-8")).items()
    }
    current = _current_violations()

    unexpected: list[str] = []
    for name, edges in current.items():
        new_edges = sorted(edges - baseline.get(name, set()))
        unexpected.extend(f"{name}: {edge}" for edge in new_edges)

    assert unexpected == []


def test_no_new_core_qt_imports():
    baseline = set(
        json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get(
            "core_qt_imports",
            [],
        )
    )
    current = {
        f"{source_module} -> {target}"
        for path in sorted((APP_DIR / "core").rglob("*.py"))
        for source_module in [_module_name(path)]
        for target in _imported_modules(path, source_module)
        if _matches_prefix(target, "PySide6")
    }

    assert sorted(current - baseline) == []


def _current_violations() -> dict[str, set[str]]:
    violations: dict[str, set[str]] = {name: set() for name in RULES}
    for path in sorted(APP_DIR.rglob("*.py")):
        source_module = _module_name(path)
        imported_modules = _imported_modules(path, source_module)
        for name, (source_prefix, target_prefix) in RULES.items():
            if not _matches_prefix(source_module, source_prefix):
                continue
            for target in imported_modules:
                if _matches_prefix(target, target_prefix):
                    violations[name].add(f"{source_module} -> {target}")
    return violations


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _imported_modules(path: Path, source_module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(source_module, node)
            if module:
                modules.add(module)
    return modules


def _resolve_import_from(source_module: str, node: ast.ImportFrom) -> str:
    if node.level <= 0:
        return node.module or ""
    module_parts = source_module.split(".")
    package_parts = module_parts[:-1]
    base_len = max(0, len(package_parts) - node.level + 1)
    target_parts = package_parts[:base_len]
    if node.module:
        target_parts.extend(node.module.split("."))
    return ".".join(target_parts)


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")
