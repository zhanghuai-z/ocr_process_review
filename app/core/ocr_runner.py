"""PaddleOCR text recognition wrapper (QThread-based).

AiStudio Serving API (confirmed):
  POST {api_url}/layout-parsing
  Authorization: token <api_token>
  Body: {"file": "<base64_JPEG>", "fileType": 1}
  Response:
    result.layoutParsingResults[0].prunedResult.overall_ocr_res:
      rec_texts:  list[str]
      rec_scores: list[float]
      rec_boxes:  list[[x_min, y_min, x_max, y_max]]
"""
from __future__ import annotations
import base64
from typing import List

from PySide6.QtCore import QThread, Signal

from app.core.bbox_utils import bbox_from_quad, sanitize_xyxy_bbox
from app.core.paddle_result_utils import (
    DEFAULT_LOCAL_LANG,
    DEFAULT_LOCAL_OCR_VERSION,
    prepare_paddle_runtime_env,
    results_to_dicts,
)
from app.core.ocr_config import get_config
from app.models import BBox, Block, BlockType, Line, Page, ProofStatus

AUTO_FLAG_THRESHOLD = 0.80


class OcrRunner:

    def __init__(self) -> None:
        self._engine = None

    # ── local mode ─────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            prepare_paddle_runtime_env()
            from paddleocr import PaddleOCR
            cfg = get_config()
            self._engine = PaddleOCR(
                lang=DEFAULT_LOCAL_LANG,
                ocr_version=cfg.get("local_ocr_version", DEFAULT_LOCAL_OCR_VERSION),
                engine="paddle_dynamic",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
        return self._engine

    def _local_ocr(self, crop_bgr) -> List[Line]:
        engine = self._get_engine()
        result = engine.predict(crop_bgr)
        lines: List[Line] = []
        crop_h, crop_w = crop_bgr.shape[:2]
        for payload in results_to_dicts(result):
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_boxes", [])
            polys = payload.get("rec_polys", [])
            for idx, text in enumerate(texts):
                score = float(scores[idx]) if idx < len(scores) else 0.0
                box_data = boxes[idx] if idx < len(boxes) else (polys[idx] if idx < len(polys) else None)
                if hasattr(box_data, "tolist"):
                    box_data = box_data.tolist()
                if isinstance(box_data, (list, tuple)) and len(box_data) >= 4:
                    if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in box_data[:4]):
                        bbox = bbox_from_quad(box_data[:4]).clamp(crop_w, crop_h)
                    else:
                        bbox = sanitize_xyxy_bbox(box_data[:4], crop_w, crop_h)
                else:
                    bbox = BBox(0, 0, 0, 0)
                if bbox.area <= 0:
                    continue
                lines.append(Line(
                    text=text,
                    confidence=float(score),
                    bbox=bbox,
                    proof_status=(
                        ProofStatus.AUTO_FLAGGED
                        if score < AUTO_FLAG_THRESHOLD
                        else ProofStatus.UNCHECKED
                    ),
                ))
        return lines

    # ── api mode ───────────────────────────────────────────────

    def _api_ocr(self, crop_bgr) -> List[Line]:
        """Call AiStudio /layout-parsing; raises on network/auth errors."""
        import cv2
        import requests
        cfg = get_config()
        url = cfg["api_url"].rstrip("/") + "/layout-parsing"
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")

        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return []
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

        # result.layoutParsingResults[0].prunedResult.overall_ocr_res
        layout_results = data.get("result", {}).get("layoutParsingResults", [])
        lines: List[Line] = []
        for item in layout_results:
            ocr_res = (
                item.get("prunedResult", {})
                    .get("overall_ocr_res", {})
            )
            texts  = ocr_res.get("rec_texts",  [])
            scores = ocr_res.get("rec_scores", [])
            boxes  = ocr_res.get("rec_boxes",  [])
            for idx, text in enumerate(texts):
                score = float(scores[idx]) if idx < len(scores) else 0.0
                if idx < len(boxes):
                    b = boxes[idx]
                    lx, ly = int(b[0]), int(b[1])
                    lw, lh = int(b[2]) - lx, int(b[3]) - ly
                else:
                    lx = ly = lw = lh = 0
                lines.append(Line(
                    text=text,
                    confidence=score,
                    bbox=BBox(lx, ly, lw, lh),
                    proof_status=(
                        ProofStatus.AUTO_FLAGGED
                        if score < AUTO_FLAG_THRESHOLD
                        else ProofStatus.UNCHECKED
                    ),
                ))
        return lines

    # ── common interface ───────────────────────────────────────

    def _run_ocr(self, crop_bgr) -> List[Line]:
        if get_config()["mode"] == "api":
            return self._api_ocr(crop_bgr)
        return self._local_ocr(crop_bgr)

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
