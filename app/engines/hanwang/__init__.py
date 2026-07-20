"""Hanwang native CharOCR integration.

``native_bridge`` owns process execution and caching. ``micro_recblock``
accepts the typed CharOCR request and returns immutable OCR observations.
Layout analysis and routing remain outside this package.
"""

from app.engines.hanwang.paths import (
    HanwangNativeAssetsMissing,
    get_hanwang_bin_dir,
    verify_hanwang_assets,
)

__all__ = [
    "HanwangNativeAssetsMissing",
    "get_hanwang_bin_dir",
    "verify_hanwang_assets",
]
