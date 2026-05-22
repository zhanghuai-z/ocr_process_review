"""HanwangLayoutEngine：实现 LayoutEngine 协议，调 doc_seg.dll 切版面区域。"""
from __future__ import annotations

from typing import List

import cv2

from app.core.logging import get_logger
from app.engines.hanwang.native_bridge import HanwangNativeError, run_docseg
from app.engines.hanwang.translator import translate_docseg
from app.models import Block

logger = get_logger(__name__)


class HanwangLayoutEngine:
    """汉王原生版面分析引擎（doc_seg.dll）。"""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self._timeout = timeout

    def analyze(self, image_path: str) -> List[Block]:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Hanwang layout: cannot read image %s", image_path)
            return []
        try:
            raw = run_docseg(image, timeout=self._timeout)
        except HanwangNativeError as e:
            logger.warning("Hanwang layout failed on %s: %s", image_path, e)
            return []
        blocks = translate_docseg(raw)
        logger.info("Hanwang layout %s → %d blocks", image_path, len(blocks))
        return blocks
