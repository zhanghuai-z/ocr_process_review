"""Legacy OCR runner wrapper used by older worker paths."""
from __future__ import annotations
from typing import List

from PySide6.QtCore import QThread, Signal

from app.engines import OcrContext
from app.engines.real_ocr_adapter import ApiOcrEngine, LocalOcrEngine
from app.core.ocr_config import get_config
from app.models import BBox, Block, BlockType, Line, Page


class OcrRunner:

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            if get_config()["mode"] == "api":
                self._engine = ApiOcrEngine()
            else:
                self._engine = LocalOcrEngine()
        return self._engine

    def _run_ocr(self, crop_bgr) -> List[Line]:
        return self._get_engine().recognize(crop_bgr, OcrContext())

    def recognize_block(self, block: Block, page_image_path: str) -> Block:
        import cv2
        if block.block_type not in (
            BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE
        ):
            return block
        img = cv2.imread(page_image_path)
        if img is None:
            return block
        bb = block.bbox
        crop = img[bb.y: bb.y + bb.h, bb.x: bb.x + bb.w]
        if crop.size == 0:
            return block
        lines = self._run_ocr(crop)
        for line in lines:
            line.bbox = BBox(
                line.bbox.x + bb.x,
                line.bbox.y + bb.y,
                line.bbox.w,
                line.bbox.h,
            )
        block.lines = lines
        return block

    def recognize_page(self, page: Page) -> Page:
        for block in page.blocks:
            self.recognize_block(block, page.image_path)
        return page


class OcrWorker(QThread):
    page_done = Signal(int, int)
    all_done  = Signal(list)
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages  = pages
        self._runner = OcrRunner()

    def run(self) -> None:
        try:
            total = len(self._pages)
            for i, page in enumerate(self._pages):
                self._runner.recognize_page(page)
                self.page_done.emit(i, total)
            self.all_done.emit(self._pages)
        except Exception as e:
            self.error.emit(str(e))
