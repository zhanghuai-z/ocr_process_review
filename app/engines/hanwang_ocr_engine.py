"""HanwangOcrEngine：实现 OcrEngine 协议，输出 crop 局部坐标。

工作流（与 hanwang_native_workflow.run_full_recognition_areas 对齐）：
1. linecut_segimg(crop, recblock=[(0,0,w,h)]) → linecut.lines（每个 area）.groups（每行）
2. 对每个 group，按 chars union + margin 求行级 bbox，sub-crop
3. linecut_recogimg(line_crop, recblock=full_line_crop) → 单行 recog
4. translate_linecut → 取 Line，把 bbox 平移回 area crop 坐标空间
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from app.core.logging import get_logger
from app.core.ocr_ir import OcrIrLine
from app.core.token_char_mapper import build_line_from_ir
from app.engines import OCR_BBOX_SPACE_CROP, OcrContext
from app.engines.hanwang.native_bridge import (
    HanwangNativeError,
    RECOG_MODE_SIMPLIFIED,
    RECOG_POSTPROCESS_DEFAULT,
    run_linecut_recog,
    run_linecut_segimg,
)
from app.engines.hanwang.translator import AUTO_FLAG_THRESHOLD, translate_linecut_ir
from app.models import BBox, Line, ProofStatus

logger = get_logger(__name__)


_LINE_MARGIN: int = 12


def _union_line_bbox(group: dict, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """对齐 hanwang_native_workflow.char_union_bbox：行 bbox = chars union + margin，clamp。"""
    chars = group.get("chars") or []
    if chars:
        ls = [int(c["bbox"]["left"]) for c in chars]
        ts = [int(c["bbox"]["top"]) for c in chars]
        rs = [int(c["bbox"]["right"]) for c in chars]
        bs = [int(c["bbox"]["bottom"]) for c in chars]
        l, t, r, b = min(ls), min(ts), max(rs), max(bs)
    else:
        bb = group.get("bbox") or {}
        l = int(bb.get("left", 0))
        t = int(bb.get("top", 0))
        r = int(bb.get("right", image_w - 1))
        b = int(bb.get("bottom", image_h - 1))
    l = max(0, l - _LINE_MARGIN)
    t = max(0, t - _LINE_MARGIN)
    r = min(image_w - 1, r + _LINE_MARGIN)
    b = min(image_h - 1, b + _LINE_MARGIN)
    return l, t, r, b


def _offset_ir_line(ir_line: OcrIrLine, dx: int, dy: int) -> OcrIrLine:
    tokens = []
    for token in ir_line.tokens:
        bbox = token.bbox
        tokens.append(
            replace(
                token,
                bbox=(
                    BBox(x=bbox.x + dx, y=bbox.y + dy, w=bbox.w, h=bbox.h)
                    if bbox is not None
                    else None
                ),
            )
        )
    return OcrIrLine(
        text=ir_line.text,
        confidence=ir_line.confidence,
        bbox=BBox(
            x=ir_line.bbox.x + dx,
            y=ir_line.bbox.y + dy,
            w=ir_line.bbox.w,
            h=ir_line.bbox.h,
        ),
        source_text=ir_line.source_text,
        tokens=tokens,
        review_flags=list(ir_line.review_flags),
    )


def _line_status(ir_line: OcrIrLine) -> ProofStatus:
    return (
        ProofStatus.AUTO_FLAGGED
        if ir_line.confidence < AUTO_FLAG_THRESHOLD
        else ProofStatus.UNCHECKED
    )


class HanwangOcrEngine:
    """汉王原生 OCR 引擎（linecut SegImg + Recog，可选 IntegratRcg CharRcg）。"""

    bbox_space: str = OCR_BBOX_SPACE_CROP

    def __init__(
        self,
        *,
        with_charrcg: bool = True,
        mode: int = RECOG_MODE_SIMPLIFIED,
        postprocess: int = RECOG_POSTPROCESS_DEFAULT,
        seg_timeout: float = 60.0,
        recog_timeout: float = 120.0,
    ) -> None:
        self._with_charrcg = with_charrcg
        self._mode = mode
        self._postprocess = postprocess
        self._seg_timeout = seg_timeout
        self._recog_timeout = recog_timeout

    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> List[Line]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        try:
            seg = run_linecut_segimg(image_bgr, timeout=self._seg_timeout)
        except HanwangNativeError as e:
            logger.warning(
                "Hanwang segimg failed page=%s block=%s: %s",
                context.page_number,
                context.block_id,
                e,
            )
            raise RuntimeError(f"Hanwang segimg failed: {e}") from e

        all_lines: List[Line] = []
        fallback_chars_recovered = 0
        fallback_lines_failed = 0
        # linecut["lines"] 是按 recblock 聚合的 area；这里只有一个 recblock=full crop
        for area in seg.get("lines", []):
            for group in area.get("groups", []):
                l, t, r, b = _union_line_bbox(group, w, h)
                if r <= l or b <= t:
                    continue
                sub_crop = image_bgr[t:b + 1, l:r + 1]
                if sub_crop.size == 0:
                    continue
                try:
                    raw = run_linecut_recog(
                        sub_crop,
                        with_charrcg=self._with_charrcg,
                        mode=self._mode,
                        postprocess=self._postprocess,
                        timeout=self._recog_timeout,
                    )
                except HanwangNativeError as e:
                    # 与 hanwang_native_workflow.run_full_recognition_areas 对齐：
                    # 行级 crop 触发 AccessViolation 时，逐字符 crop 兜底
                    logger.warning(
                        "Hanwang line recog failed page=%s block=%s bbox=(%d,%d,%d,%d): %s; "
                        "falling back to per-char crops",
                        context.page_number, context.block_id, l, t, r, b, e,
                    )
                    recovered = self._char_fallback(
                        image_bgr, group, w, h, all_lines, context,
                    )
                    fallback_chars_recovered += recovered
                    if recovered == 0:
                        fallback_lines_failed += 1
                    continue
                for ir_line in translate_linecut_ir(raw):
                    shifted = _offset_ir_line(ir_line, l, t)
                    all_lines.append(build_line_from_ir(shifted, proof_status=_line_status(shifted)))

        logger.info(
            "Hanwang OCR page=%s block=%s lines=%d "
            "(char_fallback_recovered=%d, line_fallback_failed=%d)",
            context.page_number, context.block_id, len(all_lines),
            fallback_chars_recovered, fallback_lines_failed,
        )
        return all_lines

    def _char_fallback(
        self,
        image_bgr: np.ndarray,
        group: dict,
        crop_w: int,
        crop_h: int,
        all_lines: List[Line],
        context: OcrContext,
    ) -> int:
        """逐字符 crop 兜底：对 group.chars 每个字单独 crop+recog，每个成功的字
        生成一个单字 Line。返回回收到的字符数。

        与 hanwang_native_workflow.run_full_recognition_areas 的 char fallback 对齐。
        """
        chars = group.get("chars") or []
        if not chars:
            return 0
        recovered = 0
        for ch_idx, char_raw in enumerate(chars):
            char_bbox = char_raw.get("bbox") or {}
            try:
                cl = int(char_bbox.get("left", 0))
                ct = int(char_bbox.get("top", 0))
                cr = int(char_bbox.get("right", 0))
                cb = int(char_bbox.get("bottom", 0))
            except (TypeError, ValueError):
                continue
            cl = max(0, cl - _LINE_MARGIN)
            ct = max(0, ct - _LINE_MARGIN)
            cr = min(crop_w - 1, cr + _LINE_MARGIN)
            cb = min(crop_h - 1, cb + _LINE_MARGIN)
            if cr <= cl or cb <= ct:
                continue
            sub = image_bgr[ct:cb + 1, cl:cr + 1]
            if sub.size == 0:
                continue
            try:
                raw = run_linecut_recog(
                    sub,
                    with_charrcg=self._with_charrcg,
                    mode=self._mode,
                    postprocess=self._postprocess,
                    timeout=self._recog_timeout,
                )
            except HanwangNativeError as e:
                logger.warning(
                    "Hanwang char fallback failed page=%s block=%s char_idx=%d "
                    "bbox=(%d,%d,%d,%d): %s",
                    context.page_number, context.block_id, ch_idx, cl, ct, cr, cb, e,
                )
                continue
            for ir_line in translate_linecut_ir(
                raw,
                bbox_source="hanwang:CharRcg:char_fallback",
                review_flags=["hanwang_char_fallback"],
            ):
                shifted = _offset_ir_line(ir_line, cl, ct)
                line = build_line_from_ir(shifted, proof_status=_line_status(shifted))
                all_lines.append(line)
                recovered += sum(1 for c in line.chars if c.char)
        return recovered
