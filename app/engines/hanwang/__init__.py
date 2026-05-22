"""汉王 OCR 原生引擎集成包。

层次：
- paths.py          解析 resources/hanwang_native/ 资产位置
- native_bridge.py  对 *_probe.exe 的 subprocess 封装（32 位 native 桥接）
- translator.py     原始 JSON → app.models 转换

对外暴露的引擎适配器位于：
- app/engines/hanwang_ocr_engine.py     (OcrEngine 协议)
- app/engines/hanwang_layout_engine.py  (LayoutEngine 协议)
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
