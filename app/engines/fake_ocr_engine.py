"""Fake OCR 引擎：用于测试和离线开发。

行为稳定一致，不依赖外网或 PaddleOCR。
"""
from __future__ import annotations
from typing import List

import numpy as np

from app.engines import OcrContext
from app.models import BBox, Line, ProofStatus


class FakeOcrEngine:
    """Fake OCR 引擎。

    返回固定文本行，根据输入图像尺寸生成合理的 bbox。
    """

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        """返回模拟的识别结果。"""
        h, w = image_bgr.shape[:2]

        # 根据输入尺寸模拟行数
        num_lines = max(1, h // 60)

        lines = []
        for i in range(num_lines):
            y_offset = 10 + i * 55
            if y_offset + 20 > h:
                break

            text = "测试OCR文本" if i % 2 == 0 else "低置信文本"
            confidence = 0.95 if i % 2 == 0 else 0.60

            proof = (
                ProofStatus.AUTO_FLAGGED if confidence < 0.80
                else ProofStatus.UNCHECKED
            )

            line = Line(
                text=text,
                original_text=text,
                ocr_text=text,
                confidence=confidence,
                bbox=BBox(
                    x=10,
                    y=y_offset,
                    w=min(w - 20, 200),
                    h=20,
                ),
            )
            line.set_proof_status(proof)
            lines.append(line)

        return lines
