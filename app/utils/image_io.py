"""Unicode-safe OpenCV image file access."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_cv_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image file without OpenCV's path encoding limitations.

    Windows ``cv2.imread`` may reject a valid path when an ancestor directory
    contains non-ASCII characters. NumPy opens the file path, then OpenCV only
    decodes the resulting bytes.
    """
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


__all__ = ["read_cv_image"]
