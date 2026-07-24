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
    "api_model_profile": "",       # 固定链路会按 endpoint 推断 profile
    "layout_concurrency": 8,        # 远端版面分析 job 并发数；本地 Paddle 仍保持串行
    "ocr_page_concurrency": 2,      # CharOCR 页级并发；4 为平衡档，5-20 仅供压力测试
    "paddle_api_network_mode": "auto",  # "auto" | "env_proxy" | "direct"
    "layout_debug_artifacts": False, # 开发调试时才写 Paddle raw json / overlay 图片

    # 校对
    "auto_flag_threshold": 0.80,
    "auto_save_interval": 60,      # 秒

    # UI 偏好
    "theme": "light",
    "export_default_format": "txt",

    # 自动收紧
    "tighten_padding": 6,
    "tighten_min_content_ratio": 0.002,
    "tighten_max_shrink_ratio": 0.85,

    # 校对质量评测（quality probe）
    # 这些键供 SamplerConfig.from_app_config 读取。改完无需重启即生效。
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def get_config() -> dict[str, Any]:
    """Return the runtime OCR config snapshot used by engines and UI."""
    from app.core.api_profiles import normalize_api_base_url

    cfg = AppConfig.instance()
    try:
        layout_concurrency = int(cfg.get("layout_concurrency", 8))
    except (TypeError, ValueError):
        layout_concurrency = 8
    try:
        ocr_page_concurrency = int(cfg.get("ocr_page_concurrency", 2))
    except (TypeError, ValueError):
        ocr_page_concurrency = 2
    ocr_page_concurrency = max(1, min(20, ocr_page_concurrency))
    network_mode = str(cfg.get("paddle_api_network_mode", "auto") or "auto").strip().lower()
    if network_mode not in {"auto", "env_proxy", "direct"}:
        network_mode = "auto"
    return {
        "mode": cfg.get("ocr_mode", "local"),
        "api_url": normalize_api_base_url(cfg.get("api_url", "")),
        "api_timeout": int(cfg.get("api_timeout", 30)),
        "api_token": cfg.get("api_token", ""),
        "api_layout_model_name": cfg.get("api_layout_model_name", ""),
        "api_model_profile": cfg.get("api_model_profile", ""),
        "layout_concurrency": layout_concurrency,
        "ocr_page_concurrency": ocr_page_concurrency,
        "paddle_api_network_mode": network_mode,
        "layout_debug_artifacts": _as_bool(cfg.get("layout_debug_artifacts", False)),
    }


def update_config(**kwargs: Any) -> None:
    """Update runtime OCR settings through AppConfig keys."""
    cfg = AppConfig.instance()
    mapping = {
        "mode": "ocr_mode",
        "api_model_profile": "api_model_profile",
        "api_url": "api_url",
        "api_timeout": "api_timeout",
        "api_token": "api_token",
        "api_layout_model_name": "api_layout_model_name",
        "layout_concurrency": "layout_concurrency",
        "ocr_page_concurrency": "ocr_page_concurrency",
        "paddle_api_network_mode": "paddle_api_network_mode",
        "layout_debug_artifacts": "layout_debug_artifacts",
    }
    for k, v in kwargs.items():
        if k in mapping:
            if k == "api_url":
                from app.core.api_profiles import normalize_api_base_url
                v = normalize_api_base_url(v)
            cfg.set(mapping[k], v)
