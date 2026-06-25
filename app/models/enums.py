"""全局枚举定义。"""
from enum import Enum, auto


class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    REFERENCE = "reference"
    EQUATION = "equation"
    UNKNOWN = "unknown"

    @classmethod
    def from_paddle(cls, label: str) -> "BlockType":
        """将 PaddleOCR-VL 标签映射到程序内粗粒度块类型。"""
        if not label:
            return cls.UNKNOWN

        normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "text": cls.TEXT,
            "paragraph_text": cls.TEXT,
            "text_block": cls.TEXT,
            "text_box": cls.TEXT,
            "body": cls.TEXT,
            "title": cls.TITLE,
            "doc_title": cls.TITLE,
            "doc_heading": cls.TITLE,
            "section_title": cls.TITLE,
            "chapter_title": cls.TITLE,
            "paragraph_title": cls.TITLE,
            "text_title": cls.TITLE,
            "heading": cls.TITLE,
            "headline": cls.TITLE,
            "heading_1": cls.TITLE,
            "heading_2": cls.TITLE,
            "heading_3": cls.TITLE,
            "heading_4": cls.TITLE,
            "heading_5": cls.TITLE,
            "heading_6": cls.TITLE,
            "paragraph": cls.TEXT,
            "plain_text": cls.TEXT,
            "doc_text": cls.TEXT,
            "text_region": cls.TEXT,
            "body_text": cls.TEXT,
            "content": cls.TEXT,
            "abstract": cls.TEXT,
            "table_of_contents": cls.TEXT,
            "toc": cls.TEXT,
            "page_number": cls.TEXT,
            "number": cls.TEXT,
            "header": cls.TEXT,
            "footer": cls.TEXT,
            "footnote": cls.TEXT,
            "vision_footnote": cls.TEXT,
            "sidebar_text": cls.TEXT,
            "algorithm": cls.TEXT,
            "figure": cls.FIGURE,
            "graphic": cls.FIGURE,
            "photo": cls.FIGURE,
            "logo": cls.FIGURE,
            "image": cls.FIGURE,
            "picture": cls.FIGURE,
            "illustration": cls.FIGURE,
            "chart": cls.FIGURE,
            "seal": cls.FIGURE,
            "header_image": cls.FIGURE,
            "footer_image": cls.FIGURE,
            "figure_caption": cls.FIGURE_CAPTION,
            "figure_title": cls.FIGURE_CAPTION,
            "caption": cls.FIGURE_CAPTION,
            "figure_note": cls.FIGURE_CAPTION,
            "image_caption": cls.FIGURE_CAPTION,
            "table": cls.TABLE,
            "table_region": cls.TABLE,
            "table_block": cls.TABLE,
            "table_cell": cls.TABLE,
            "table_body": cls.TABLE,
            "table_caption": cls.TABLE_CAPTION,
            "table_caption_text": cls.TABLE_CAPTION,
            "table_note": cls.TABLE_CAPTION,
            "table_title": cls.TABLE_CAPTION,
            "reference": cls.REFERENCE,
            "reference_content": cls.REFERENCE,
            "references": cls.REFERENCE,
            "reference_list": cls.REFERENCE,
            "reference_text": cls.REFERENCE,
            "bibliography": cls.REFERENCE,
            "equation": cls.EQUATION,
            "equation_block": cls.EQUATION,
            "isolated_formula": cls.EQUATION,
            "inline_formula": cls.EQUATION,
            "formula": cls.EQUATION,
            "formula_number": cls.EQUATION,
        }
        if normalized in mapping:
            return mapping[normalized]

        if "title" in normalized or normalized.startswith("heading_"):
            return cls.TITLE
        if "caption" in normalized and "table" in normalized:
            return cls.TABLE_CAPTION
        if "note" in normalized and "table" in normalized:
            return cls.TABLE_CAPTION
        if "caption" in normalized:
            return cls.FIGURE_CAPTION
        if "table" in normalized:
            return cls.TABLE
        if any(token in normalized for token in ("reference", "bibliography")):
            return cls.REFERENCE
        if any(token in normalized for token in ("paragraph", "text", "body", "content")):
            return cls.TEXT
        if any(token in normalized for token in ("figure", "image", "picture", "illustration", "graphic", "logo", "photo")):
            return cls.FIGURE
        if any(token in normalized for token in ("equation", "formula", "math")):
            return cls.EQUATION
        return cls.UNKNOWN


class ProofStatus(str, Enum):
    UNCHECKED = "unchecked"       # 未校对
    AUTO_FLAGGED = "auto_flagged" # 自动标记（低置信度）
    MODIFIED = "modified"         # 人工已修改
    OK = "ok"                     # 确认无误


class PageStatus(str, Enum):
    """页面级别的状态枚举。"""
    IMPORTED = "imported"
    PREPROCESSED = "preprocessed"
    LAYOUT_DONE = "layout_done"
    LAYOUT_CONFIRMED = "layout_confirmed"
    OCR_DONE = "ocr_done"
    PROOFING = "proofing"
    PROOF_DONE = "proof_done"
    ERROR = "error"


class BlockSource(str, Enum):
    """块来源枚举。"""
    AUTO_LAYOUT = "auto_layout"
    MANUAL_DRAW = "manual_draw"
    AUTO_TIGHTENED = "auto_tightened"
    USER_EDITED = "user_edited"


class CanvasMode(str, Enum):
    """编辑画布交互模式。"""
    PAN = "pan"
    SELECT = "select"
    DRAW_BLOCK = "draw_block"
    EDIT_BLOCK = "edit_block"
