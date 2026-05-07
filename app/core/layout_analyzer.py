"""PP-StructureV3 layout analysis wrapper. Supports local / api modes."""
from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from app.core.api_response_utils import (
    build_api_payload,
    extract_api_markdown_text,
    get_api_result_items,
    iter_api_ocr_payloads,
    resolve_api_endpoint,
    split_markdown_paragraphs,
)
from app.core.bbox_utils import (
    bbox_from_quad,
    bbox_from_xyxy,
    sanitize_xyxy_bbox,
    scale_bbox,
    scale_bbox_to_page,
)
from app.core.logging import get_logger
from app.core.paddle_result_utils import (
    get_local_layout_init_kwargs,
    prepare_paddle_runtime_env,
    results_to_dicts,
)
from app.models import BBox, Block, BlockType, Page

logger = get_logger(__name__)
LOCAL_LAYOUT_CANVAS_W = 800
LOCAL_LAYOUT_CANVAS_H = 608


class LayoutWorker(QThread):
    """版面分析 Worker 线程，避免阻塞 UI。"""
    page_done = Signal(int, int)   # (current_index, total)
    all_done  = Signal(list)       # List[Page]
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages

    def run(self) -> None:
        analyzer = None
        try:
            analyzer = LayoutAnalyzer()
            total = len(self._pages)
            for i, page in enumerate(self._pages):
                analyzer.analyze(page)
                self.page_done.emit(i, total)
            self.all_done.emit(self._pages)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if analyzer is not None:
                analyzer.close()


class LayoutAnalyzer:

    def __init__(self) -> None:
        self._engine = None

    def close(self) -> None:
        import gc

        engine = self._engine
        self._engine = None
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        gc.collect()

    # ── local mode ─────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            prepare_paddle_runtime_env()
            from paddleocr import PPStructureV3

            init_kwargs = get_local_layout_init_kwargs()
            logger.info(
                "Initializing PPStructureV3 layout engine: profile=%s, "
                "engine=paddle_static, enable_mkldnn=False, "
                "use_doc_orientation_classify=False, use_doc_unwarping=False, "
                "use_textline_orientation=False, use_table_recognition=False, "
                "use_formula_recognition=False, use_chart_recognition=False, "
                "use_region_detection=False, format_block_content=False",
                "default",
            )
            self._engine = PPStructureV3(**init_kwargs)
        return self._engine

    def _unwrap_layout_items(self, result) -> List[dict]:
        """兼容不同 Paddle 结果包装结构。"""
        items: List[dict] = []
        for payload in results_to_dicts(result):
            if isinstance(payload.get("layout"), list):
                items.extend(item for item in payload["layout"] if isinstance(item, dict))
                continue
            if isinstance(payload.get("result"), list):
                items.extend(item for item in payload["result"] if isinstance(item, dict))
                continue
            if isinstance(payload.get("layoutParsingResults"), list):
                items.extend(item for item in payload["layoutParsingResults"] if isinstance(item, dict))
                continue
            items.append(payload)
        return items

    def _extract_layout_boxes(self, item: dict) -> List[dict]:
        for payload in (
            item,
            item.get("prunedResult", {}) if isinstance(item, dict) else {},
        ):
            if not isinstance(payload, dict):
                continue
            layout_det_res = payload.get("layout_det_res", {})
            if isinstance(layout_det_res, dict) and isinstance(layout_det_res.get("boxes"), list):
                return [box for box in layout_det_res["boxes"] if isinstance(box, dict)]
        if isinstance(item, dict) and item.get("bbox") is not None:
            return [item]
        return []

    def _extract_parsing_items(self, item: dict) -> List[dict]:
        """读取 AIStudio layout-parsing 的结构化段落块。"""
        if not isinstance(item, dict):
            return []

        candidates = (
            item,
            item.get("prunedResult", {}) if isinstance(item.get("prunedResult"), dict) else {},
        )
        for payload in candidates:
            parsing_res_list = payload.get("parsing_res_list")
            if isinstance(parsing_res_list, list):
                return [entry for entry in parsing_res_list if isinstance(entry, dict)]
        return []

    def _extract_int_pair(self, value) -> Optional[tuple[int, int]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(round(float(value[0]))), int(round(float(value[1])))
            except (TypeError, ValueError):
                return None
        return None

    def _extract_size_from_payload(self, payload) -> Optional[tuple[int, int]]:
        """从任意层级 payload 中提取坐标空间尺寸。"""
        if not isinstance(payload, dict):
            return None

        pair_keys = (
            "image_size",
            "original_image_size",
            "orig_size",
            "page_size",
            "coord_size",
            "coordinate_size",
        )
        for key in pair_keys:
            pair = self._extract_int_pair(payload.get(key))
            if pair and pair[0] > 0 and pair[1] > 0:
                return pair

        width_keys = ("width", "image_width", "imageWidth", "orig_width", "page_width")
        height_keys = ("height", "image_height", "imageHeight", "orig_height", "page_height")
        width = next((payload.get(key) for key in width_keys if payload.get(key) is not None), None)
        height = next((payload.get(key) for key in height_keys if payload.get(key) is not None), None)
        try:
            if width is not None and height is not None:
                width = int(round(float(width)))
                height = int(round(float(height)))
                if width > 0 and height > 0:
                    return width, height
        except (TypeError, ValueError):
            pass

        shape = payload.get("img_shape") or payload.get("input_shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            try:
                first = int(round(float(shape[0])))
                second = int(round(float(shape[1])))
            except (TypeError, ValueError):
                return None
            if first > 0 and second > 0:
                # img_shape 常见为 [h, w]
                return second, first
        return None

    def _resolve_coordinate_space(
        self,
        page: Page,
        *,
        data: Optional[dict] = None,
        layout_item: Optional[dict] = None,
        box_item: Optional[dict] = None,
    ) -> Optional[tuple[int, int]]:
        """优先从响应元数据中解析 bbox 所属坐标空间尺寸。"""
        for payload in (box_item, layout_item):
            size = self._extract_size_from_payload(payload)
            if size:
                return size

        if isinstance(layout_item, dict):
            pruned = layout_item.get("prunedResult", {})
            for payload in (
                pruned,
                pruned.get("layout_det_res", {}),
                pruned.get("overall_ocr_res", {}),
            ):
                size = self._extract_size_from_payload(payload)
                if size:
                    return size

        if isinstance(data, dict):
            size = self._extract_size_from_payload(data)
            if size:
                return size
            result = data.get("result", {})
            if isinstance(result, dict):
                for payload in (
                    result,
                    result.get("dataInfo", {}),
                ):
                    size = self._extract_size_from_payload(payload)
                    if size:
                        return size
        return None

    def _should_scale_bbox_to_page(
        self,
        bbox: BBox,
        page: Page,
        coordinate_space: tuple[int, int],
    ) -> bool:
        source_w, source_h = coordinate_space
        if source_w <= 0 or source_h <= 0:
            return False
        if source_w == page.width and source_h == page.height:
            return False

        tolerance = 2
        bbox_exceeds_source = bbox.x2 > source_w + tolerance or bbox.y2 > source_h + tolerance
        bbox_fits_page = (
            bbox.x1 >= -tolerance
            and bbox.y1 >= -tolerance
            and bbox.x2 <= page.width + tolerance
            and bbox.y2 <= page.height + tolerance
        )
        if bbox_exceeds_source and bbox_fits_page:
            logger.debug(
                "API bbox already fits page pixels; ignoring conflicting coordinate space "
                "bbox=%s page=%dx%d source=%dx%d",
                bbox.to_xyxy(),
                page.width,
                page.height,
                source_w,
                source_h,
            )
            return False
        return True

    def _extract_bbox_from_coordinate(
        self,
        coord,
        page: Page,
        *,
        coordinate_space: Optional[tuple[int, int]] = None,
    ):
        """兼容 API 返回的 xyxy、四点坐标、相对坐标与其他坐标空间。"""
        if not isinstance(coord, (list, tuple)):
            return None

        bbox = None
        is_relative_coord = False
        if len(coord) >= 4 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in coord[:4]):
            flat_points = coord[:4]
            try:
                is_relative_coord = all(
                    0.0 <= float(point[0]) <= 1.0 and 0.0 <= float(point[1]) <= 1.0
                    for point in flat_points
                )
            except (TypeError, ValueError):
                is_relative_coord = False
            if is_relative_coord:
                bbox = bbox_from_quad([
                    (float(point[0]) * page.width, float(point[1]) * page.height)
                    for point in flat_points
                ])
            else:
                bbox = bbox_from_quad(flat_points)
        elif len(coord) >= 4:
            values = coord[:4]
            try:
                is_relative_coord = all(0.0 <= float(value) <= 1.0 for value in values)
            except (TypeError, ValueError):
                is_relative_coord = False
            if is_relative_coord:
                bbox = bbox_from_xyxy([
                    float(values[0]) * page.width,
                    float(values[1]) * page.height,
                    float(values[2]) * page.width,
                    float(values[3]) * page.height,
                ])
            else:
                bbox = bbox_from_xyxy(values)
        if bbox is None:
            return None

        if coordinate_space and not is_relative_coord and self._should_scale_bbox_to_page(
            bbox,
            page,
            coordinate_space,
        ):
            source_w, source_h = coordinate_space
            bbox = scale_bbox_to_page(
                bbox,
                source_w=source_w,
                source_h=source_h,
                page_w=page.width,
                page_h=page.height,
            )
        else:
            bbox = bbox.clamp(page.width, page.height)
        return bbox

    def _extract_api_line_blocks(
        self,
        item: dict,
        page: Page,
        *,
        coordinate_space: Optional[tuple[int, int]] = None,
        start_order: int = 0,
    ) -> tuple[list[Block], list[tuple[str, object]]]:
        blocks: list[Block] = []
        overlay_items: list[tuple[str, object]] = []
        order = start_order

        for payload in iter_api_ocr_payloads(item):
            texts = payload.get("rec_texts", [])
            boxes = payload.get("rec_boxes", [])
            polys = payload.get("rec_polys", [])
            for idx, text in enumerate(texts):
                box_data = boxes[idx] if idx < len(boxes) else (polys[idx] if idx < len(polys) else None)
                bbox = self._extract_bbox_from_coordinate(
                    box_data,
                    page,
                    coordinate_space=coordinate_space,
                )
                if not bbox or bbox.area <= 0:
                    continue
                block_type = BlockType.TEXT
                overlay_items.append((block_type.value, bbox))
                blocks.append(Block(
                    block_type=block_type,
                    bbox=bbox,
                    order=order,
                    note=(text or "")[:200],
                ))
                order += 1

        return blocks, overlay_items

    def _union_block_bboxes(self, blocks: list[Block], page: Page) -> BBox:
        x1 = min(block.bbox.x1 for block in blocks)
        y1 = min(block.bbox.y1 for block in blocks)
        x2 = max(block.bbox.x2 for block in blocks)
        y2 = max(block.bbox.y2 for block in blocks)
        return bbox_from_xyxy([x1, y1, x2, y2]).clamp(page.width, page.height)

    def _infer_plain_text_block_type(self, text: str) -> BlockType:
        stripped = (text or "").strip()
        if not stripped:
            return BlockType.TEXT
        if stripped.startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、")):
            return BlockType.TITLE
        if stripped.startswith(("（一）", "（二）", "（三）", "（四）", "（五）", "(", "（")) and len(stripped) <= 24:
            return BlockType.TITLE
        return BlockType.TEXT

    def _merge_ocr_line_blocks_for_layout(
        self,
        line_blocks: list[Block],
        page: Page,
    ) -> list[Block]:
        """PP-OCRv5 只有文字行框，版面阶段需要聚合成更易操作的段落块。"""
        if len(line_blocks) < 6:
            return line_blocks

        sorted_blocks = sorted(line_blocks, key=lambda block: (block.bbox.y1, block.bbox.x1))
        heights = sorted(block.bbox.h for block in sorted_blocks if block.bbox.h > 0)
        if not heights:
            return line_blocks
        median_h = heights[len(heights) // 2]
        max_line_gap = max(12, int(round(median_h * 0.85)))
        page_left = min(block.bbox.x1 for block in sorted_blocks)
        indent_threshold = max(40, int(round(median_h * 1.2)))

        groups: list[list[Block]] = []
        current: list[Block] = []
        for block in sorted_blocks:
            if not current:
                current = [block]
                continue
            previous_bbox = current[-1].bbox
            vertical_gap = block.bbox.y1 - previous_bbox.y2
            starts_indented_paragraph = (
                len(current) >= 2
                and block.bbox.x1 - page_left > indent_threshold
            )
            previous_is_short_heading = (
                len(current) == 1
                and current[-1].bbox.w <= page.width * 0.45
                and vertical_gap > max(8, int(round(median_h * 0.3)))
            )
            if vertical_gap > max_line_gap or starts_indented_paragraph or previous_is_short_heading:
                groups.append(current)
                current = [block]
            else:
                current.append(block)
        if current:
            groups.append(current)

        if len(groups) == len(line_blocks):
            return line_blocks

        merged_blocks: list[Block] = []
        for order, group in enumerate(groups):
            text = "\n".join(block.note for block in group if block.note)
            merged_blocks.append(Block(
                block_type=self._infer_plain_text_block_type(text),
                bbox=self._union_block_bboxes(group, page),
                order=order,
                note=text[:200],
            ))
        return merged_blocks

    def _extract_api_parsing_blocks(
        self,
        item: dict,
        page: Page,
        *,
        coordinate_space: Optional[tuple[int, int]] = None,
        start_order: int = 0,
    ) -> tuple[list[Block], list[tuple[str, object]]]:
        parsing_items = self._extract_parsing_items(item)
        if not parsing_items:
            return [], []

        def sort_key(index_and_entry: tuple[int, dict]) -> tuple[int, int]:
            index, entry = index_and_entry
            try:
                return 0, int(entry.get("block_order"))
            except (TypeError, ValueError):
                try:
                    return 0, int(entry.get("order"))
                except (TypeError, ValueError):
                    return 1, index

        blocks: list[Block] = []
        overlay_items: list[tuple[str, object]] = []
        for offset, (_, entry) in enumerate(sorted(enumerate(parsing_items), key=sort_key)):
            coord = (
                entry.get("block_bbox")
                or entry.get("block_polygon_points")
                or entry.get("coordinate")
                or entry.get("bbox")
            )
            bbox = self._extract_bbox_from_coordinate(
                coord,
                page,
                coordinate_space=coordinate_space,
            )
            if not bbox or bbox.area <= 0:
                continue

            raw_type = (
                entry.get("block_label")
                or entry.get("label")
                or entry.get("type")
                or "unknown"
            )
            content = (
                entry.get("block_content")
                or entry.get("content")
                or entry.get("text")
                or ""
            )
            block_type = BlockType.from_paddle(raw_type)
            overlay_items.append((raw_type, bbox))
            blocks.append(Block(
                block_type=block_type,
                bbox=bbox,
                order=start_order + offset,
                note=str(content)[:200],
            ))

        return blocks, overlay_items

    def _infer_markdown_block_type(self, paragraph: str) -> BlockType:
        text = (paragraph or "").lstrip()
        if not text:
            return BlockType.UNKNOWN
        if text.startswith("#"):
            return BlockType.TITLE
        if text.startswith("|") or "\n|" in text:
            return BlockType.TABLE
        if text.startswith("!["):
            return BlockType.FIGURE
        return BlockType.TEXT

    def _build_markdown_fallback_blocks(
        self,
        item: dict,
        page: Page,
        *,
        start_order: int = 0,
    ) -> tuple[list[Block], list[tuple[str, object]]]:
        paragraphs = split_markdown_paragraphs(extract_api_markdown_text(item))
        if not paragraphs:
            return [], []

        blocks: list[Block] = []
        overlay_items: list[tuple[str, object]] = []

        margin_x = max(12, int(page.width * 0.04))
        top_margin = max(12, int(page.height * 0.04))
        bottom_margin = max(12, int(page.height * 0.04))
        gap = max(8, int(page.height * 0.015))
        available_height = max(page.height - top_margin - bottom_margin - gap * (len(paragraphs) - 1), len(paragraphs) * 24)

        weights = [max(1, paragraph.count("\n") + 1) for paragraph in paragraphs]
        total_weight = sum(weights) or len(paragraphs)
        heights = [max(24, round(available_height * weight / total_weight)) for weight in weights]

        overflow = sum(heights) - available_height
        idx = len(heights) - 1
        while overflow > 0 and idx >= 0:
            shrink = min(overflow, max(0, heights[idx] - 24))
            heights[idx] -= shrink
            overflow -= shrink
            idx -= 1

        y = top_margin
        for offset, paragraph in enumerate(paragraphs):
            height = heights[offset]
            bbox = BBox(
                margin_x,
                y,
                max(1, page.width - 2 * margin_x),
                max(24, min(height, page.height - y - bottom_margin)),
            ).clamp(page.width, page.height)
            block_type = self._infer_markdown_block_type(paragraph)
            overlay_items.append((block_type.value, bbox))
            blocks.append(Block(
                block_type=block_type,
                bbox=bbox,
                order=start_order + offset,
                note=paragraph[:200],
            ))
            y += bbox.h + gap

        return blocks, overlay_items

    def _build_api_payload(self, file_b64: str, file_type: int) -> dict:
        return build_api_payload(file_b64, file_type)

    def _write_debug_response(self, page: Page, data: object, suffix: str) -> None:
        """把布局结果落到工作图旁边，方便复盘漂移页。"""
        image_path = page.display_image_path
        if not image_path:
            return
        debug_path = Path(image_path).with_suffix(suffix)
        payload = {
            "page": {
                "display_image_path": image_path,
                "source_path": page.source_path,
                "width": page.width,
                "height": page.height,
                "page_number": page.page_number,
            },
            "response": data,
        }
        try:
            debug_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj),
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write layout debug response: %s", exc)

    def _write_api_debug_response(self, page: Page, data: dict) -> None:
        self._write_debug_response(page, data, ".layout-api.json")

    def _write_local_debug_response(self, page: Page, data: object) -> None:
        self._write_debug_response(page, data, ".layout-local.json")

    def _build_local_debug_payload(self, payloads: List[dict]) -> List[dict]:
        slim_payloads: List[dict] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            slim_payloads.append({
                "input_path": payload.get("input_path"),
                "page_index": payload.get("page_index"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "layout_det_res": payload.get("layout_det_res"),
                "parsing_res_list": payload.get("parsing_res_list"),
            })
        return slim_payloads

    def _write_bbox_overlay(
        self,
        page: Page,
        *,
        items: List[tuple[str, object]],
        suffix: str,
        color: tuple[int, int, int],
    ) -> None:
        import cv2

        image_path = page.display_image_path
        if not image_path:
            return
        img = cv2.imread(image_path)
        if img is None:
            return

        canvas = img.copy()
        for label, bbox in items:
            if bbox is None or getattr(bbox, "area", 0) <= 0:
                continue
            x1, y1, x2, y2 = bbox.to_xyxy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(
                    canvas,
                    str(label),
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out_path = Path(image_path).with_suffix(suffix)
        try:
            cv2.imwrite(str(out_path), canvas)
        except Exception as exc:
            logger.warning("Failed to write bbox overlay %s: %s", out_path, exc)

    def _rescale_blocks_if_suspicious(self, page: Page) -> None:
        """当所有框都像落在较小坐标系上时，做一次统一比例修正。"""
        valid_blocks = [block for block in page.blocks if block.bbox.area > 0]
        if len(valid_blocks) < 2 or page.width <= 0 or page.height <= 0:
            return

        max_x2 = max(block.bbox.x2 for block in valid_blocks)
        max_y2 = max(block.bbox.y2 for block in valid_blocks)
        if max_x2 <= 0 or max_y2 <= 0:
            return

        looks_like_local_model_canvas = (
            page.width > LOCAL_LAYOUT_CANVAS_W * 1.2
            and page.height > LOCAL_LAYOUT_CANVAS_H * 1.2
            and max_x2 <= LOCAL_LAYOUT_CANVAS_W * 1.05
            and max_y2 <= LOCAL_LAYOUT_CANVAS_H * 1.05
            and (max_x2 >= LOCAL_LAYOUT_CANVAS_W * 0.72 or max_y2 >= LOCAL_LAYOUT_CANVAS_H * 0.72)
        )
        if looks_like_local_model_canvas:
            scale_x = page.width / LOCAL_LAYOUT_CANVAS_W
            scale_y = page.height / LOCAL_LAYOUT_CANVAS_H
            logger.warning(
                "Layout bbox coordinates look like local model canvas (%dx%d); "
                "rescaling for %s with scale_x=%.3f scale_y=%.3f",
                LOCAL_LAYOUT_CANVAS_W,
                LOCAL_LAYOUT_CANVAS_H,
                page.display_image_path,
                scale_x,
                scale_y,
            )
            for block in page.blocks:
                block.bbox = scale_bbox(block.bbox, scale_x, scale_y).clamp(page.width, page.height)
            return

        scale_x = page.width / max_x2
        scale_y = page.height / max_y2
        ratio_gap = max(scale_x, scale_y) / max(min(scale_x, scale_y), 1e-6)

        suspicious_uniform_scale = (
            scale_x >= 1.2
            and scale_y >= 1.2
            and scale_x <= 6.0
            and scale_y <= 6.0
            and ratio_gap <= 1.2
            and (max_x2 >= page.width * 0.35 or max_y2 >= page.height * 0.35)
        )
        if not suspicious_uniform_scale:
            return

        logger.warning(
            "Layout bbox coordinates look scaled-down; applying uniform rescale "
            "for %s: scale_x=%.3f scale_y=%.3f page=%dx%d max_bbox=(%d,%d)",
            page.display_image_path, scale_x, scale_y, page.width, page.height, max_x2, max_y2,
        )
        for block in page.blocks:
            block.bbox = scale_bbox(block.bbox, scale_x, scale_y).clamp(page.width, page.height)

    def _local_analyze(self, page: Page) -> Page:
        from PIL import Image

        engine = self._get_engine()
        try:
            with Image.open(page.display_image_path) as image:
                page.width, page.height = image.size
        except OSError as exc:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}") from exc
        result = engine.predict(page.display_image_path)
        raw_payloads = results_to_dicts(result)
        self._write_local_debug_response(page, self._build_local_debug_payload(raw_payloads))
        items = self._unwrap_layout_items(raw_payloads)
        page.blocks = []
        raw_overlay_items: List[tuple[str, object]] = []
        order = 0
        for item in items:
            coordinate_space = self._resolve_coordinate_space(page, data=item, layout_item=item)
            layout_boxes = self._extract_layout_boxes(item)
            for box in layout_boxes:
                raw_type = box.get("label") or box.get("type") or item.get("type") or item.get("label") or "unknown"
                bbox_raw = box.get("coordinate") or box.get("bbox") or item.get("bbox", [0, 0, 0, 0])
                bbox = self._extract_bbox_from_coordinate(
                    bbox_raw,
                    page,
                    coordinate_space=self._resolve_coordinate_space(
                        page,
                        data=item,
                        layout_item=item,
                        box_item=box,
                    ) or coordinate_space,
                )
                if not bbox or bbox.area <= 0:
                    continue
                raw_overlay_items.append((raw_type, bbox))
                page.blocks.append(Block(
                    block_type=BlockType.from_paddle(raw_type),
                    bbox=bbox,
                    order=order,
                ))
                order += 1
        del raw_payloads
        del items
        self._write_bbox_overlay(
            page,
            items=raw_overlay_items,
            suffix=".layout-local-raw.png",
            color=(0, 165, 255),
        )
        self._rescale_blocks_if_suspicious(page)
        self._write_bbox_overlay(
            page,
            items=[(block.block_type.value, block.bbox) for block in page.blocks],
            suffix=".layout-local-app-overlay.png",
            color=(80, 220, 80),
        )
        return page

    # ── api mode ───────────────────────────────────────────────

    def _api_analyze(self, page: Page) -> Page:
        """Call AiStudio OCR/layout endpoint; raises on network/auth errors."""
        import cv2
        import requests
        from app.core.ocr_config import get_config

        cfg = get_config()
        url = resolve_api_endpoint(cfg["api_url"], default_suffix="/layout-parsing")
        timeout = cfg["api_timeout"]
        token = cfg.get("api_token", "")

        img = cv2.imread(page.display_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {page.display_image_path}")
        page.height, page.width = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError(f"Cannot encode image: {page.display_image_path}")
        file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        resp = requests.post(
            url,
            json=self._build_api_payload(file_b64, 1),
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._write_api_debug_response(page, data)

        result_items = get_api_result_items(data)
        page.blocks = []
        raw_overlay_items: List[tuple[str, object]] = []
        order = 0
        for item in result_items:
            boxes = (
                item.get("prunedResult", {})
                    .get("layout_det_res", {})
                    .get("boxes", [])
            )
            coordinate_space = self._resolve_coordinate_space(
                page,
                data=data,
                layout_item=item,
            )

            parsing_blocks, parsing_overlay_items = self._extract_api_parsing_blocks(
                item,
                page,
                coordinate_space=coordinate_space,
                start_order=order,
            )
            if parsing_blocks:
                page.blocks.extend(parsing_blocks)
                raw_overlay_items.extend(parsing_overlay_items)
                order += len(parsing_blocks)
                continue

            if boxes:
                for box in boxes:
                    coord = box.get("coordinate", [0, 0, 0, 0])
                    bbox = self._extract_bbox_from_coordinate(
                        coord,
                        page,
                        coordinate_space=coordinate_space,
                    )
                    if not bbox or bbox.area <= 0:
                        continue
                    raw_type = box.get("label", "unknown")
                    raw_overlay_items.append((raw_type, bbox))
                    page.blocks.append(Block(
                        block_type=BlockType.from_paddle(raw_type),
                        bbox=bbox,
                        order=order,
                    ))
                    order += 1
                continue

            line_blocks, line_overlay_items = self._extract_api_line_blocks(
                item,
                page,
                coordinate_space=coordinate_space,
                start_order=order,
            )
            if line_blocks:
                line_blocks = self._merge_ocr_line_blocks_for_layout(line_blocks, page)
                line_overlay_items = [(block.block_type.value, block.bbox) for block in line_blocks]
                page.blocks.extend(line_blocks)
                raw_overlay_items.extend(line_overlay_items)
                order += len(line_blocks)
                continue

            markdown_blocks, markdown_overlay_items = self._build_markdown_fallback_blocks(
                item,
                page,
                start_order=order,
            )
            if markdown_blocks:
                page.blocks.extend(markdown_blocks)
                raw_overlay_items.extend(markdown_overlay_items)
                order += len(markdown_blocks)
        self._write_bbox_overlay(
            page,
            items=raw_overlay_items,
            suffix=".layout-api-raw.png",
            color=(0, 165, 255),
        )
        self._rescale_blocks_if_suspicious(page)
        self._write_bbox_overlay(
            page,
            items=[(block.block_type.value, block.bbox) for block in page.blocks],
            suffix=".layout-app-overlay.png",
            color=(80, 220, 80),
        )
        return page

    # ── common interface ───────────────────────────────────────

    def analyze(self, page: Page) -> Page:
        from app.core.ocr_config import get_config
        if get_config()["mode"] == "api":
            return self._api_analyze(page)
        return self._local_analyze(page)

    def analyze_pages(self, pages: List[Page]) -> List[Page]:
        return [self.analyze(page) for page in pages]
