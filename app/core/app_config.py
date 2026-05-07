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
    "ocr_mode": "local",           # "local" | "api" | "mock"
    "api_url": "",
    "api_timeout": 30,
    "api_token": "",
    "api_layout_model_name": "",
    "api_model_profile": "",   # 选中的官方预设 key，空 = 未选/自定义

    # LLM 预审
    "llm_pre_review_enabled": False,
    "llm_provider": "fake",        # "fake" | "http" | "openai-compatible" | "local"
    "llm_endpoint": "",
    "llm_model": "",
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


# 兼容旧代码的快速访问
def get_config() -> dict[str, Any]:
    """兼容旧 OCR 配置接口，返回 dict。"""
    cfg = AppConfig.instance()
    return {
        "mode": cfg.get("ocr_mode", "local"),
        "api_url": cfg.get("api_url", ""),
        "api_timeout": int(cfg.get("api_timeout", 30)),
        "api_token": cfg.get("api_token", ""),
        "api_layout_model_name": cfg.get("api_layout_model_name", ""),
        "api_model_profile": cfg.get("api_model_profile", ""),
    }


def update_config(**kwargs: Any) -> None:
    """兼容旧 OCR 配置更新接口。"""
    cfg = AppConfig.instance()
    mapping = {
        "mode": "ocr_mode",
        "api_url": "api_url",
        "api_timeout": "api_timeout",
        "api_token": "api_token",
        "api_layout_model_name": "api_layout_model_name",
        "api_model_profile": "api_model_profile",
    }
    for k, v in kwargs.items():
        if k in mapping:
            cfg.set(mapping[k], v)
