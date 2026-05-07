"""兼容旧接口的 OCR 配置桥接层。

历史代码直接从本模块读取 OCR 配置，但当前正式配置已经迁移到
`app.core.app_config`（QSettings 持久化）。

保留本模块是为了兼容旧调用路径，避免 UI 设置和真实 OCR pipeline
读取两套互不相通的配置。
"""
from __future__ import annotations

from app.core.app_config import get_config as _get_config
from app.core.app_config import update_config as _update_config


def get_config() -> dict:
    return _get_config()


def update_config(**kwargs) -> None:
    _update_config(**kwargs)
