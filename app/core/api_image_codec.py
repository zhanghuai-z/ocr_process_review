"""Image encoding helpers for Paddle/AiStudio API uploads."""
from __future__ import annotations

import base64

import cv2
import numpy as np

PADDLE_API_IMAGE_EXT = ".png"
PADDLE_API_PNG_COMPRESSION = 3


def encode_image_bytes_for_paddle(image_bgr: np.ndarray) -> bytes:
    """Encode an image losslessly for Paddle image API uploads."""
    ok, buf = cv2.imencode(
        PADDLE_API_IMAGE_EXT,
        image_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), PADDLE_API_PNG_COMPRESSION],
    )
    if not ok:
        return b""
    return buf.tobytes()


def encode_image_b64_for_paddle(image_bgr: np.ndarray) -> str:
    """Encode an image losslessly for Paddle image API payloads."""
    image_bytes = encode_image_bytes_for_paddle(image_bgr)
    if not image_bytes:
        return ""
    return base64.b64encode(image_bytes).decode("ascii")
