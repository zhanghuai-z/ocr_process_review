"""Ratchet direct dependencies from the UI package to lower layers."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_UI = ROOT / "app" / "ui"
BASELINE_PATH = Path(__file__).with_name("ui_dependency_baseline.json")


def test_ui_has_no_new_lower_layer_dependencies() -> None:
    baseline, prefixes = _load_baseline()
    current = _current_edges(prefixes)
    unexpected = sorted(current - baseline)

    assert unexpected == [], (
        "new app.ui direct dependencies detected; either remove the import or "
        "make an intentional architecture change and update the migration plan: "
        + ", ".join(unexpected)
    )


def _load_baseline() -> tuple[set[str], tuple[str, ...]]:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("UI dependency baseline must be a JSON object")
    prefixes = data.get("forbidden_target_prefixes")
    existing = data.get("existing_violations")
    if not isinstance(prefixes, list) or not all(
        isinstance(item, str) and item for item in prefixes
    ):
        raise TypeError("UI dependency baseline prefixes must be non-empty strings")
    if not isinstance(existing, list) or not all(
        isinstance(item, str) and item for item in existing
    ):
        raise TypeError("UI dependency baseline edges must be non-empty strings")
    return set(existing), tuple(prefixes)


def _current_edges(prefixes: tuple[str, ...]) -> set[str]:
    edges: set[str] = set()
    for path in sorted(APP_UI.rglob("*.py")):
        source = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for target in _import_targets(source, path, node):
                if _matches_prefix(target, prefixes):
                    edges.add(f"{source} -> {target}")
    return edges


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_targets(source: str, path: Path, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level == 0:
        return (node.module or "",)

    source_parts = source.split(".")
    package_parts = (
        source_parts
        if path.name == "__init__.py"
        else source_parts[:-1]
    )
    base_len = max(0, len(package_parts) - node.level + 1)
    target_parts = package_parts[:base_len]
    if node.module:
        target_parts.extend(node.module.split("."))
    return (".".join(target_parts),)


def _matches_prefix(target: str, prefixes: tuple[str, ...]) -> bool:
    return any(target == prefix or target.startswith(prefix + ".") for prefix in prefixes)
