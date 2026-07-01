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


class OcrPolicy(str, Enum):
    TEXT_OCR = "text_ocr"
    PRESERVE_AS_FORMULA = "preserve_as_formula"
    PRESERVE_AS_TABLE = "preserve_as_table"
    SKIP = "skip"
    MANUAL_ONLY = "manual_only"


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
