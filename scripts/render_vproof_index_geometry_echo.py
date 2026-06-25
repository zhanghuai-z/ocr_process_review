from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
from app.services.char_index_service import CharIndexService
from app.services.proof_reference_context import build_proof_reference_context


DEFAULT_OUT_DIR = Path("debug/vproof_index_geometry_echo")


def _draw_bbox(
    image,
    bbox: BBox,
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 2,
    label_pos: tuple[int, int] | None = None,
) -> None:
    cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            label_pos or (bbox.x, max(14, bbox.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def render_vproof_index_geometry_echo(out_dir: Path | str = DEFAULT_OUT_DIR) -> dict[str, Any]:
    """Render a VProof geometry echo for a line containing a multi-char token.

    The important invariant is that CharEntry.char_idx is a display-text
    offset, not a line.chars array index. If this regresses, the boxes still
    look correct in layout/Hanwang data, but VProof text selection shifts after
    the inline formula carrier.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    carrier = "$ Incentive_{c} \\times Post_{t} $"
    image = np.full((260, 620, 3), 255, dtype=np.uint8)
    line = Line(
        text=f"A+{carrier}税后",
        confidence=0.9,
        bbox=BBox(16, 120, 560, 46),
        chars=[
            Char(char="A", confidence=0.9, bbox=BBox(24, 132, 14, 26), bbox_source="ocr", bbox_granularity="char", token_text="A"),
            Char(char="+", confidence=0.9, bbox=BBox(44, 132, 12, 26), bbox_source="ocr", bbox_granularity="char", token_text="+"),
            Char(char=carrier, confidence=0.9, bbox=BBox(78, 132, 300, 26), bbox_source="paddle_inline_formula", bbox_granularity="word", token_text=carrier),
            Char(char="税", confidence=0.9, bbox=BBox(430, 132, 30, 26), bbox_source="hanwang:CharRcg", bbox_granularity="char", token_text="税"),
            Char(char="后", confidence=0.9, bbox=BBox(510, 132, 30, 26), bbox_source="hanwang:CharRcg", bbox_granularity="char", token_text="后"),
        ],
    )

    for char in line.chars:
        if char.bbox is not None:
            cv2.rectangle(image, (char.bbox.x + 2, char.bbox.y + 6), (char.bbox.x2 - 2, char.bbox.y2 - 6), (35, 35, 35), -1)

    page_path = out_dir / "synthetic_vproof_page.png"
    cv2.imwrite(str(page_path), image)

    page = Page(
        image_path=str(page_path),
        width=image.shape[1],
        height=image.shape[0],
        page_number=1,
        blocks=[Block(block_type=BlockType.TEXT, order=0, bbox=BBox(10, 112, 588, 64), lines=[line])],
    )
    project = OcrProject(name="vproof-index-geometry-echo", pages=[page])
    svc = CharIndexService(include_non_cjk=True, include_fallback=True).build_index(project)
    reference_context = build_proof_reference_context(page)
    flat_text = reference_context.text
    text_map = reference_context.slots

    carrier_entry = svc.first_entry(carrier)
    tax_entry = svc.first_entry("税")
    after_entry = svc.first_entry("后")
    if carrier_entry is None or tax_entry is None or after_entry is None:
        raise AssertionError("missing expected VProof index entries")

    overlay = image.copy()
    cv2.putText(overlay, "VProof index geometry echo", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(overlay, "Green=line, Orange=line.chars, Magenta=formula token", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(overlay, "Red/Blue=trailing CJK slots after the formula token", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)
    _draw_bbox(overlay, line.bbox, (0, 180, 0), "line", thickness=2, label_pos=(18, 110))
    for index, char in enumerate(line.chars):
        if char.bbox is not None:
            label_y = 184 if index in (0, 2) else 202
            _draw_bbox(overlay, char.bbox, (0, 150, 255), f"char[{index}]", thickness=1, label_pos=(char.bbox.x, label_y))
    _draw_bbox(overlay, carrier_entry.bbox, (255, 0, 255), f"formula display_idx={carrier_entry.char_idx}", thickness=2, label_pos=(78, 110))
    _draw_bbox(overlay, tax_entry.bbox, (0, 0, 255), f"tax display_idx={tax_entry.char_idx}", thickness=2, label_pos=(390, 224))
    _draw_bbox(overlay, after_entry.bbox, (255, 0, 0), f"after display_idx={after_entry.char_idx}", thickness=2, label_pos=(390, 244))

    overlay_path = out_dir / "vproof_index_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    tax_text_pos = text_map[tax_entry.char_idx].start
    after_text_pos = text_map[after_entry.char_idx].start
    report_path = out_dir / "vproof_index_geometry_echo.md"
    report_path.write_text(
        "\n".join([
            "# VProof index geometry echo",
            "",
            "Legend:",
            "- Green: line bbox.",
            "- Orange: source line.chars bbox.",
            "- Magenta: inline formula carrier index entry.",
            "- Red/blue: trailing CJK entries after the carrier.",
            "",
            "Invariant:",
            "- `CharEntry.char_idx` is the display-text offset.",
            "- A multi-character formula carrier occupies one `line.chars` slot but many display-text positions.",
            "- Entries after that carrier must point to their real display-text offsets.",
            "",
            f"carrier display_idx: {carrier_entry.char_idx}",
            f"tax display_idx: {tax_entry.char_idx}, text_map char: {flat_text[tax_text_pos]}",
            f"after display_idx: {after_entry.char_idx}, text_map char: {flat_text[after_text_pos]}",
            f"overlay: {overlay_path}",
            "",
        ]),
        encoding="utf-8",
    )

    return {
        "carrier": carrier,
        "flat_text": flat_text,
        "text_map": text_map,
        "carrier_entry": carrier_entry,
        "tax_entry": tax_entry,
        "after_entry": after_entry,
        "overlay_path": overlay_path,
        "report_path": report_path,
    }


def main() -> None:
    result = render_vproof_index_geometry_echo()
    print(result["report_path"])
    print(result["overlay_path"])


if __name__ == "__main__":
    main()
