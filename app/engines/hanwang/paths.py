"""汉王原生资产路径解析与完整性校验。

资产位于 ``<repo>/resources/hanwang_native/bin``。
打包后（PyInstaller）由 spec 文件落到 ``sys._MEIPASS/resources/hanwang_native/bin``。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


# 运行 hanwang 必须存在的最小文件清单（基于 hanwang_native_workflow 实测依赖）。
REQUIRED_BINARIES: tuple[str, ...] = (
    "linecut_segimg_probe.exe",
    "linecut_recogimg_probe.exe",
    "docseg_probe.exe",
)
REQUIRED_DLLS: tuple[str, ...] = (
    "linecut.dll",
    "IntegratRcg.dll",
    "doc_seg.dll",
    "mp30.dll",
    "mp60.dll",
)


class HanwangNativeAssetsMissing(RuntimeError):
    """资产目录不存在 / 缺少关键文件。"""


def _candidate_roots() -> Iterable[Path]:
    """按优先级返回 hanwang_native 根目录候选位置。

    顺序：
    1. 环境变量 HANWANG_NATIVE_DIR（运维覆盖，可指向 hanwang_native 或 bin）
    2. PyInstaller 解包目录 sys._MEIPASS/resources/hanwang_native
    3. 仓库默认位置：本文件向上 4 层 + resources/hanwang_native
    """
    env_override = os.environ.get("HANWANG_NATIVE_DIR")
    if env_override:
        env_path = Path(env_override)
        yield env_path.parent if env_path.name.lower() == "bin" else env_path

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        yield Path(meipass) / "resources" / "hanwang_native"

    # app/engines/hanwang/paths.py → ../../.. = repository root
    repo_default = Path(__file__).resolve().parents[3] / "resources" / "hanwang_native"
    yield repo_default


def get_hanwang_root() -> Path:
    """返回第一个真实存在的资产根目录；找不到时抛 HanwangNativeAssetsMissing。"""
    for candidate in _candidate_roots():
        if candidate.is_dir():
            return candidate
    raise HanwangNativeAssetsMissing(
        "未找到 hanwang_native 资产目录。请确认 "
        "resources/hanwang_native/ 已就位，"
        "或设置环境变量 HANWANG_NATIVE_DIR 指向 hanwang_native 或 bin。"
    )


def get_hanwang_bin_dir() -> Path:
    """资产 bin 子目录，即所有 .exe / .dll 所在位置。"""
    return get_hanwang_root() / "bin"


def verify_hanwang_assets() -> None:
    """启动期完整性校验。缺失则抛 HanwangNativeAssetsMissing。"""
    bin_dir = get_hanwang_bin_dir()
    if not bin_dir.is_dir():
        raise HanwangNativeAssetsMissing(f"bin 目录不存在: {bin_dir}")
    missing: list[str] = []
    for name in REQUIRED_BINARIES + REQUIRED_DLLS:
        if not (bin_dir / name).is_file():
            missing.append(name)
    if missing:
        raise HanwangNativeAssetsMissing(
            f"hanwang_native/bin 缺少关键文件: {', '.join(missing)} (位于 {bin_dir})"
        )
