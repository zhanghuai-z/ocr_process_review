"""Stable scope identities for derived OCR observations."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from app.models.layout_snapshot import LayoutSnapshot


def layout_snapshot_fingerprint(snapshot: LayoutSnapshot) -> str:
    payload = {
        "page_uid": snapshot.page_uid,
        "artifact_uid": snapshot.artifact_uid,
        "blocks": [
            {
                "uid": block.uid,
                "bbox": list(block.bbox.to_xyxy()),
                "type": block.block_type.value,
                "source_label": block.source_label,
                "order": block.order,
                "ocr_policy": block.ocr_policy.value,
            }
            for block in snapshot.blocks
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def page_image_hash(image_bgr: np.ndarray) -> str:
    if image_bgr.ndim < 2:
        raise ValueError("page image hash requires an image array")
    contiguous = np.ascontiguousarray(image_bgr)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


__all__ = ["layout_snapshot_fingerprint", "page_image_hash"]
