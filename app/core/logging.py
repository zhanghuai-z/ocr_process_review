"""统一日志模块。

GUI 弹窗显示人能读懂的信息，日志文件保存技术细节。
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path


_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """初始化根日志器。

    Args:
        log_dir: 日志目录，默认取项目下 logs/ 目录。
        level: 日志级别。
        log_to_file: 是否写入文件。

    Returns:
        根日志器。
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台 handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(console)

    # 文件 handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / "ocr_process.log", encoding="utf-8", mode="a"
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取子日志器。"""
    return logging.getLogger(name)


# 用户友好的错误提示
USER_ERROR_MAP: dict[str, str] = {
    "Image file not found": "图片文件找不到，请检查路径。",
    "PDF render failed": "PDF 渲染失败，文件可能已损坏。",
    "OCR engine error": "OCR 引擎出错，请检查配置或重试。",
    "Network error": "网络连接失败，请检查网络和 API 地址。",
    "API timeout": "API 请求超时，请稍后重试。",
    "Invalid API response": "API 返回格式异常，请检查后端状态。",
    "Layout analysis failed": "版面分析失败，请检查图片或重试。",
    "Permission denied": "文件权限不足，请检查文件访问权限。",
    "Disk full": "磁盘空间不足，请清理后重试。",
    "Export failed": "导出失败，请检查导出设置。",
}

# Schema 版本
SCHEMA_VERSION = 25
APP_VERSION = "0.2.0"
