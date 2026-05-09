"""Snapshot debug helpers."""
from __future__ import annotations
import json
from pathlib import Path
from tools.ocr_inspector.models.ir import DocumentNode


def save_snapshot(doc: DocumentNode, out_path: str) -> None:
    payload = {
        "engine": doc.engine,
        "source_path": doc.source_path,
        "parse_log": doc.parse_log,
        "pages": [_page_to_dict(p) for p in doc.pages],
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox_dict(b):
    return b.to_dict() if b else None


def _page_to_dict(page):
    return {
        "page_number": page.page_number,
        "image_path": page.image_path,
        "blocks": [_block_to_dict(b) for b in page.blocks],
        "orphan_lines": [_line_to_dict(ln) for ln in page.orphan_lines],
    }


def _block_to_dict(block):
    return {
        "id": block.id, "label": block.label, "bbox": _bbox_dict(block.bbox),
        "content": block.content, "order": block.order,
        "lines": [_line_to_dict(ln) for ln in block.lines],
    }


def _line_to_dict(line):
    return {
        "id": line.id, "text": line.text, "confidence": line.confidence,
        "bbox": _bbox_dict(line.bbox), "source_field": line.source_field,
        "chars": [_char_to_dict(c) for c in line.chars],
    }


def _char_to_dict(char):
    return {
        "id": char.id, "char": char.char, "token_text": char.token_text,
        "bbox": _bbox_dict(char.bbox), "confidence": char.confidence,
        "kind": char.kind, "bbox_source": char.bbox_source,
        "bbox_granularity": char.bbox_granularity,
    }
