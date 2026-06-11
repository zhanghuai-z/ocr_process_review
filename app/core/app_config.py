"""配置管理。

全局配置通过 QSettings 持久化，项目级配置存 SQLite。
"""
from __future__ import annotations
import os
from typing import Any

from PySide6.QtCore import QSettings


# 默认配置
_DEFAULT_CONFIG: dict[str, Any] = {
    # OCR 引擎
    "ocr_mode": "local",           # 默认保留本地 Paddle；API/汉王作为可选引擎
    "api_url": "",
    "api_timeout": 30,
    "api_token": "",
    "api_layout_model_name": "",
    "api_model_profile": "",       # 历史配置字段；固定链路会按 endpoint 推断

    # LLM 预审
    "llm_pre_review_enabled": False,
    "llm_provider": "fake",        # "fake" | "http" | "openai-compatible" | "local"
    "llm_endpoint": "",
    "llm_api_key": "",
    "llm_model": "",
    "llm_rules_path": "",
    "llm_timeout": 30,
    "llm_batch_lines": 30,
    "llm_send_context": True,
    "llm_send_page_text_only": True,

    # 校对
    "auto_flag_threshold": 0.80,
    "auto_save_interval": 60,      # 秒

    # UI 偏好
    "theme": "dark_teal",
    "export_default_format": "txt",

    # 自动收紧
    "tighten_padding": 6,
    "tighten_min_content_ratio": 0.002,
    "tighten_max_shrink_ratio": 0.85,

    # 校对质量评测（quality probe）
    # 这些键供 SamplerConfig.from_app_config 读取。改完无需重启即生效。
    "quality_probe_sand_count": 25,        # 每 quality_probe_sand_unit_chars 个可切图字符投放几个沙子
    "quality_probe_sand_unit_chars": 1000, # 1000=每千字；10000=每万字
    "quality_probe_target_ratio": 0.025,   # 旧配置兼容：未设置 sand_count 时使用
    "quality_probe_min_total": 8,
    "quality_probe_max_total": 35,
    "quality_probe_max_per_page": 2,
    "quality_probe_max_per_line": 1,
    "quality_probe_auto_enable": True,     # 打开有 OCR 数据的项目时是否自动启用评测
}


class AppConfig:
    """全局应用配置，基于 QSettings。"""

    _instance: AppConfig | None = None

    def __init__(self) -> None:
        self._settings = QSettings("ocr_process", "ocr_process")

    @classmethod
    def instance(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, key: str, default: Any = None) -> Any:
        """读取配置，优先 QSettings，其次默认值，最后环境变量。"""
        # 环境变量覆盖
        env_key = f"OCR_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

        # QSettings
        qval = self._settings.value(key)
        if qval is not None:
            return qval

        # 默认值
        return _DEFAULT_CONFIG.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)

    def get_all(self) -> dict[str, Any]:
        """获取完整配置快照（用于 UI 展示）。"""
        result = {}
        for key in _DEFAULT_CONFIG:
            result[key] = self.get(key)
        return result

    def reset_to_defaults(self) -> None:
        self._settings.clear()


def get_config() -> dict[str, Any]:
    """Return the runtime OCR/LLM config snapshot used by engines and UI."""
    from app.core.api_profiles import normalize_api_base_url

    cfg = AppConfig.instance()
    return {
        "mode": cfg.get("ocr_mode", "local"),
        "api_url": normalize_api_base_url(cfg.get("api_url", "")),
        "api_timeout": int(cfg.get("api_timeout", 30)),
        "api_token": cfg.get("api_token", ""),
        "api_layout_model_name": cfg.get("api_layout_model_name", ""),
        "api_model_profile": cfg.get("api_model_profile", ""),
        "llm_endpoint": cfg.get("llm_endpoint", ""),
        "llm_api_key": cfg.get("llm_api_key", ""),
        "llm_rules_path": cfg.get("llm_rules_path", ""),
    }


def update_config(**kwargs: Any) -> None:
    """Update runtime OCR/LLM settings through AppConfig keys."""
    cfg = AppConfig.instance()
    mapping = {
        "mode": "ocr_mode",
        "api_model_profile": "api_model_profile",
        "api_url": "api_url",
        "api_timeout": "api_timeout",
        "api_token": "api_token",
        "api_layout_model_name": "api_layout_model_name",
        "llm_endpoint": "llm_endpoint",
        "llm_api_key": "llm_api_key",
        "llm_rules_path": "llm_rules_path",
    }
    for k, v in kwargs.items():
        if k in mapping:
            if k == "api_url":
                from app.core.api_profiles import normalize_api_base_url
                v = normalize_api_base_url(v)
            cfg.set(mapping[k], v)
