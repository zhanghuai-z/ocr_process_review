"""PaddleOCR adapter helpers."""

from .layout_importer import map_paddle_label_to_block_type
from .ppocr_v6_prepass import (
    PPOCR_V6_MODEL,
    PpOcrV6LineHint,
    PpOcrV6PrepassArtifact,
    PpOcrV6PrepassClient,
    PpOcrV6WordBox,
    build_ppocr_v6_prepass_options,
    parse_ppocr_v6_prepass_jsonl,
    parse_ppocr_v6_prepass_result,
)

__all__ = [
    "PPOCR_V6_MODEL",
    "PpOcrV6LineHint",
    "PpOcrV6PrepassArtifact",
    "PpOcrV6PrepassClient",
    "PpOcrV6WordBox",
    "build_ppocr_v6_prepass_options",
    "map_paddle_label_to_block_type",
    "parse_ppocr_v6_prepass_jsonl",
    "parse_ppocr_v6_prepass_result",
]
