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
    "ocr_mode": "api",             # 当前主程序固定使用 API 双模型链
    "api_url": "",
    "api_timeout": 30,
    "api_token": "",
    "api_layout_model_name": "",
    "api_model_profile": "",       # UI 不再暴露模型选择；保留字段仅兼容旧配置

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
    "quality_probe_target_ratio": 0.025,   # 投放比例 2.5%
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


# 兼容旧代码的快速访问
def get_config() -> dict[str, Any]:
    """兼容旧 OCR 配置接口，返回 dict。"""
    cfg = AppConfig.instance()
    return {
        "mode": cfg.get("ocr_mode", "api"),
        "api_url": cfg.get("api_url", ""),
        "api_timeout": int(cfg.get("api_timeout", 30)),
        "api_token": cfg.get("api_token", ""),
        "api_layout_model_name": cfg.get("api_layout_model_name", ""),
        "api_model_profile": cfg.get("api_model_profile", ""),
        "llm_endpoint": cfg.get("llm_endpoint", ""),
        "llm_api_key": cfg.get("llm_api_key", ""),
        "llm_rules_path": cfg.get("llm_rules_path", ""),
    }


def update_config(**kwargs: Any) -> None:
    """兼容旧 OCR 配置更新接口。"""
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
            cfg.set(mapping[k], v)
