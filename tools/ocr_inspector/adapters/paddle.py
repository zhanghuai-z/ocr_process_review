"""PaddleOCR adapter: thin wrapper around the replaceable core seam."""
from __future__ import annotations

from typing import Any

from tools.ocr_inspector.adapters.base import BaseAdapter
from tools.ocr_inspector.core import build_paddle_document
from tools.ocr_inspector.models.ir import DocumentNode


class PaddleAdapter(BaseAdapter):
    @classmethod
    def detect(cls, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        result = raw.get("result", raw)
        return isinstance(result, dict) and (
            "overall_ocr_res" in result
            or "parsing_res_list" in result
            or "rec_texts" in result
            or "layoutParsingResults" in result
            or "ocrResults" in result
        )

    def parse(self, raw: Any, *, source_path: str = "", image_path: str = "") -> DocumentNode:
        return build_paddle_document(raw, source_path=source_path, image_path=image_path).document
