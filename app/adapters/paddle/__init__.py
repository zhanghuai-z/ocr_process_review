"""PaddleOCR adapter helpers."""

from .layout_importer import map_paddle_label_to_block_type
from .ppocr_v6_prepass import (
    PPOCR_V6_MODEL,
    PpOcrV6LineHint,
    PpOcrV6PrepassArtifact,
    PpOcrV6PrepassClient,
    PpOcrV6RoutingRequestProfile,
    PpOcrV6WordBox,
    build_ppocr_v6_routing_request_profile,
    normalize_ppocr_v6_prepass_result,
    parse_ppocr_v6_prepass_jsonl,
)

__all__ = [
    "PPOCR_V6_MODEL",
    "PpOcrV6LineHint",
    "PpOcrV6PrepassArtifact",
    "PpOcrV6PrepassClient",
    "PpOcrV6RoutingRequestProfile",
    "PpOcrV6WordBox",
    "build_ppocr_v6_routing_request_profile",
    "map_paddle_label_to_block_type",
    "normalize_ppocr_v6_prepass_result",
    "parse_ppocr_v6_prepass_jsonl",
]
