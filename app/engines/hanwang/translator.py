"""把原生 JSON 翻译成 app.models (Block / Line / Char)。

约定：
- linecut_recog 返回的 bbox 是 crop 局部坐标；
  按 OcrEngine 协议 (engines/__init__.py: OCR_BBOX_SPACE_CROP)，调用方负责转回 page 坐标。
- docseg areas 是 page 坐标（来自整页图）；翻译为 LayoutEngine 输出的 Block。
- chars[].codes 是 little-endian GBK 编码整数（与 decode_gbk_word 保持一致）；
  scores 越**小**越好（probe 透传 mp30 原始分），转 0~1 置信度时取 1 - score/100，clamp 到 [0,1]。
"""
from __future__ import annotations

from typing import Iterable, List

from app.core.ocr_ir import OcrIrLine, OcrIrToken, OcrRun, build_ocr_run, classify_ir_text
from app.core.token_char_mapper import build_line_from_ir
from app.models import BBox, Block, BlockSource, BlockType, Line, ProofStatus


# 与 real_ocr_adapter.AUTO_FLAG_THRESHOLD 保持一致
AUTO_FLAG_THRESHOLD: float = 0.80


def decode_gbk_word(code: int) -> str:
    """little-endian GBK 整数 → 单字符串；解不出来返回空串。"""
    if not code:
        return ""
    raw = bytes([code]) if code <= 0x7F else bytes([code & 0xFF, (code >> 8) & 0xFF])
    try:
        text = raw.decode("gbk")
    except UnicodeDecodeError:
        return ""
    # 过滤控制字符
    return "".join(c for c in text if c in ("\t", "\n", "\r") or ord(c) >= 0x20)


def _score_to_confidence(score: int) -> float:
    """mp30 score 越小越好；线性映射到 [0,1]。"""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return 0.0
    conf = 1.0 - v / 100.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def _bbox_from_raw(raw: dict) -> BBox:
    """probe 的 bbox dict {left,top,right,bottom,width,height} → app.BBox(x,y,w,h)。

    raw 缺字段时退化为零矩形，避免抛异常断流。
    """
    left = int(raw.get("left", 0))
    top = int(raw.get("top", 0))
    width = int(raw.get("width", max(0, int(raw.get("right", left)) - left + 1)))
    height = int(raw.get("height", max(0, int(raw.get("bottom", top)) - top + 1)))
    return BBox(x=left, y=top, w=max(0, width), h=max(0, height))


def _translate_char_token(raw: dict, *, row_index: int, token_index: int, bbox_source: str) -> OcrIrToken:
    codes: list = raw.get("codes") or []
    scores: list = raw.get("scores") or []
    top_code = codes[0] if codes else 0
    top_score = scores[0] if scores else 100
    # codes 可能是字符串（如 "0xfdb9"）或整数
    if isinstance(top_code, str):
        top_code_int = int(top_code, 0) if top_code.startswith(("0x", "0X")) else int(top_code, 16)
    else:
        top_code_int = int(top_code)
    text = decode_gbk_word(top_code_int)
    return OcrIrToken(
        text=text,
        bbox=_bbox_from_raw(raw.get("bbox") or {}),
        row_index=row_index,
        token_index=token_index,
        raw_region=raw,
        confidence=_score_to_confidence(top_score),
        kind=classify_ir_text(text),
        bbox_source=bbox_source,
        bbox_granularity="char",
    )


def _iter_groups(raw_lines: Iterable[dict]) -> Iterable[dict]:
    """linecut JSON 的 lines[].groups[] 才是“一行可识别文本”的真实单位。"""
    for ln in raw_lines or []:
        for g in ln.get("groups") or []:
            yield g


def translate_linecut_ir(
    raw: dict,
    *,
    bbox_source: str = "hanwang:CharRcg",
    review_flags: list[str] | None = None,
) -> List[OcrIrLine]:
    """linecut_recog raw JSON → List[OcrIrLine]（坐标系：crop-local）。

    每个 group 翻译成一个 Line：
    - text 拼接所有字符 top-1
    - confidence 取所有字符 top-1 置信度的均值
    - tokens 保留字符级 bbox + 候选 top-1
    """
    out: List[OcrIrLine] = []
    for row_index, g in enumerate(_iter_groups(raw.get("lines", []))):
        raw_chars = g.get("chars") or []
        tokens = [
            _translate_char_token(c, row_index=row_index, token_index=token_index, bbox_source=bbox_source)
            for token_index, c in enumerate(raw_chars)
        ]
        text = "".join(token.text for token in tokens)
        if raw_chars:
            confidences = [
                _score_to_confidence((c.get("scores") or [100])[0])
                for c in raw_chars
            ]
            avg_conf = sum(confidences) / len(confidences)
        else:
            avg_conf = 0.0
        bbox = _bbox_from_raw(g.get("bbox") or {})
        if bbox.w <= 0 or bbox.h <= 0:
            continue
        out.append(
            OcrIrLine(
                text=text,
                confidence=avg_conf,
                bbox=bbox,
                source_text=text,
                tokens=tokens,
                review_flags=list(review_flags or []),
            )
        )
    return out


def translate_linecut_run(
    raw: dict,
    *,
    page_uid: str = "",
    block_uid: str = "",
    engine_version: str = "native",
    input_layout_revision: int = 0,
    bbox_source: str = "hanwang:CharRcg",
    review_flags: list[str] | None = None,
) -> OcrRun:
    return build_ocr_run(
        engine="hanwang",
        engine_version=engine_version,
        page_uid=page_uid,
        block_uid=block_uid,
        input_layout_revision=input_layout_revision,
        lines=translate_linecut_ir(
            raw,
            bbox_source=bbox_source,
            review_flags=review_flags,
        ),
    )


def translate_linecut(raw: dict) -> List[Line]:
    """Compatibility adapter: linecut_recog raw JSON → List[Line] via OCR_IR."""
    out = []
    for ir_line in translate_linecut_ir(raw):
        out.append(
            build_line_from_ir(
                ir_line,
                proof_status=(
                    ProofStatus.AUTO_FLAGGED
                    if ir_line.confidence < AUTO_FLAG_THRESHOLD
                    else ProofStatus.UNCHECKED
                ),
            )
        )
    return out


def translate_docseg(raw: dict) -> List[Block]:
    """docseg raw JSON → List[Block]（坐标系：page）。

    doc_seg.dll 的 type 字段在已观测样本中均为 2（文本类）；映射到 BlockType.TEXT。
    其他 type 值出现时仍当 TEXT 处理（保守，不丢内容），同时写到 note 备查。
    """
    blocks: List[Block] = []
    for idx, area in enumerate(raw.get("areas") or []):
        bbox = _bbox_from_raw(area)
        if bbox.w <= 0 or bbox.h <= 0:
            continue
        area_type = int(area.get("type", -1))
        blocks.append(
            Block(
                block_type=BlockType.TEXT,
                bbox=bbox,
                order=idx,
                source=BlockSource.AUTO_LAYOUT,
                note=f"hanwang doc_seg type={area_type}",
            )
        )
    return blocks


__all__ = [
    "AUTO_FLAG_THRESHOLD",
    "decode_gbk_word",
    "translate_docseg",
    "translate_linecut_ir",
    "translate_linecut_run",
    "translate_linecut",
]
