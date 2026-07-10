"""Parse and offset the native EngCut character payload."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class EngcutChar:
    text: str
    bbox: XYXY | None = None
    code: int = 0
    line_index: int = 0
    group_index: int = 0
    char_index: int = 0


def engcut_chars_from_payload(payload: dict[str, Any]) -> list[EngcutChar]:
    """Return character observations from the EngCut native JSON payload."""
    chars: list[EngcutChar] = []
    for line_index, line in enumerate(payload.get("lines") or []):
        if not isinstance(line, dict):
            continue
        for group_index, group in enumerate(line.get("groups") or []):
            if not isinstance(group, dict):
                continue
            for char_index, char in enumerate(group.get("chars") or []):
                if not isinstance(char, dict):
                    continue
                codes = char.get("codes") or []
                try:
                    code = _code_to_int(codes[0]) if codes else 0
                except (TypeError, ValueError):
                    code = 0
                chars.append(
                    EngcutChar(
                        text=_code_to_text(code),
                        bbox=_bbox_from_raw(char.get("bbox")),
                        code=code,
                        line_index=line_index,
                        group_index=group_index,
                        char_index=int(char.get("index", char_index) or 0),
                    )
                )
    return chars


def offset_engcut_chars(chars: Iterable[EngcutChar], *, dx: int, dy: int) -> list[EngcutChar]:
    """Translate crop-local EngCut geometry back to page coordinates."""
    shifted: list[EngcutChar] = []
    for char in chars:
        bbox = None
        if char.bbox is not None:
            x1, y1, x2, y2 = char.bbox
            bbox = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        shifted.append(
            EngcutChar(
                text=char.text,
                bbox=bbox,
                code=char.code,
                line_index=char.line_index,
                group_index=char.group_index,
                char_index=char.char_index,
            )
        )
    return shifted


def _code_to_text(code: int) -> str:
    return chr(code) if 32 <= code <= 126 else "~"


def _code_to_int(code: object) -> int:
    if isinstance(code, str):
        return int(code, 0) if code.startswith(("0x", "0X")) else int(code, 16)
    return int(code)


def _bbox_from_raw(raw: object) -> XYXY | None:
    if not isinstance(raw, dict):
        return None
    try:
        left = int(raw.get("left", raw.get("x", 0)))
        top = int(raw.get("top", raw.get("y", 0)))
        if "right" in raw and "bottom" in raw:
            right = int(raw["right"])
            bottom = int(raw["bottom"])
        else:
            right = left + int(raw.get("width", 0))
            bottom = top + int(raw.get("height", 0))
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
