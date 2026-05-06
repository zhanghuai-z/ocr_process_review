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
        """将 PP-Structure 返回的 type 字符串映射到枚举。"""
        mapping = {
            "text": cls.TEXT,
            "title": cls.TITLE,
            "figure": cls.FIGURE,
            "figure_caption": cls.FIGURE_CAPTION,
            "table": cls.TABLE,
            "table_caption": cls.TABLE_CAPTION,
            "reference": cls.REFERENCE,
            "equation": cls.EQUATION,
        }
        return mapping.get(label.lower(), cls.UNKNOWN)


class ProofStatus(str, Enum):
    UNCHECKED = "unchecked"       # 未校对
    AUTO_FLAGGED = "auto_flagged" # 自动标记（低置信度）
    MODIFIED = "modified"         # 人工已修改
    OK = "ok"                     # 确认无误
