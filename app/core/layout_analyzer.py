"""PP-StructureV3 layout analysis wrapper. Supports local / api modes.

AiStudio Serving API (confirmed):
  POST {api_url}/layout-parsing
  Authorization: token <api_token>
  Body: {"file": "<base64_JPEG>", "fileType": 1}
  Response:
    result.layoutParsingResults[0].prunedResult:
      layout_det_res.boxes[]:
        label:      str  (e.g. "doc_title", "paragraph", ...)
        coordinate: [x1, y1, x2, y2]  (float, page coords)
"""
from __future__ import annotations
import base64
from typing import List

from PySide6.QtCore import QThread, Signal

from app.models import BBox, Block, BlockType, Page


class LayoutWorker(QThread):
    """版面分析 Worker 线程，避免阻塞 UI。"""
    page_done = Signal(int, int)   # (current_index, total)
    all_done  = Signal(list)       # List[Page]
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages

    def run(self) -> None:
        try:
            analyzer = LayoutAnalyzer()
            total = len(self._pages)
            for i, page in enumerate(self._pages):
                analyzer.analyze(page)
                self.page_done.emit(i, total)
            self.all_done.emit(self._pages)
        except Exception as e:
            self.error.emit(str(e))


class LayoutAnalyzer:

    def __init__(self) -> None:
        self._engine = None

    # ── local mode ─────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                use_angle_cls=False,
                lang="ch",
                show_log=False,
                layout=True,
                table=False,
                ocr=False,
            )
        return self._engine

    def _local_analyze(self, page: Page) -> Page:
        engine = self._get_engine()
        result = engine.predict(page.image_path)
        page.blocks = []
        for i, item in enumerate(result or []):
            raw_type = item.get("type", "unknown")
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            page.blocks.append(Block(
                block_type=BlockType.from_paddle(raw_type),
                bbox=BBox.from_xyxy(x1, y1, x2, y2),
                order=i,
            ))
        return page

    # ── api mode ───────────────────────────────────────────────

    def _api_analyze(self, page: Page) -> Page:
        """Call AiStudio /layout-parsing; raises on network/auth errors."""
        import cv2
        import requests
        from app.core.ocr_config import get_config

        cfg = get_config()
        url = cfg["api_url"].rstrip("/") + "/layout-parsing"
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")

        img = cv2.imread(page.image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.image_path}")
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError(f"Cannot encode image: {page.image_path}")
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = requests.post(
            url,
            json={"file": file_b64, "fileType": 1},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # result.layoutParsingResults[0].prunedResult.layout_det_res.boxes
        layout_results = data.get("result", {}).get("layoutParsingResults", [])
        page.blocks = []
        order = 0
        for item in layout_results:
            boxes = (
                item.get("prunedResult", {})
                    .get("layout_det_res", {})
                    .get("boxes", [])
            )
            for box in boxes:
                coord = box.get("coordinate", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in coord]
                raw_type = box.get("label", "unknown")
                page.blocks.append(Block(
                    block_type=BlockType.from_paddle(raw_type),
                    bbox=BBox.from_xyxy(x1, y1, x2, y2),
                    order=order,
                ))
                order += 1
        return page

    # ── common interface ───────────────────────────────────────

    def analyze(self, page: Page) -> Page:
        from app.core.ocr_config import get_config
        if get_config()["mode"] == "api":
            return self._api_analyze(page)
        return self._local_analyze(page)

    def analyze_pages(self, pages: List[Page]) -> List[Page]:
        return [self.analyze(page) for page in pages]
