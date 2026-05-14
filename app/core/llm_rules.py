"""LLM candidate rules file loading."""
from __future__ import annotations

from pathlib import Path

from app.core.app_config import AppConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = REPO_ROOT / "resources" / "llm_rules" / "default_rules.txt"


def get_default_llm_rules_path() -> Path:
    return DEFAULT_RULES_PATH


def resolve_llm_rules_path(path: str | None = None) -> Path:
    configured = path
    if configured is None:
        configured = str(AppConfig.instance().get("llm_rules_path", "") or "")
    configured = configured.strip()
    if not configured:
        return DEFAULT_RULES_PATH
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def load_llm_rules(path: str | None = None) -> str:
    rules_path = resolve_llm_rules_path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"LLM rules file not found: {rules_path}")
    return rules_path.read_text(encoding="utf-8")
