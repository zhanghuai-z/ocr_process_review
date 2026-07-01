"""Stable business identifiers for domain entities."""
from __future__ import annotations

import secrets
import time


_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ALLOWED_KINDS = frozenset({
    "page", "block", "line", "char", "rawocr", "layoutedit", "ocrrun",
})


def _encode_crockford32(value: int, length: int) -> str:
    chars = ["0"] * length
    for index in range(length - 1, -1, -1):
        chars[index] = _CROCKFORD32[value & 0x1F]
        value >>= 5
    return "".join(chars)


def new_ulid() -> str:
    """Return a 26-character ULID string without adding a dependency."""
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = secrets.randbits(80)
    return (
        _encode_crockford32(timestamp_ms, 10)
        + _encode_crockford32(randomness, 16)
    )


def new_entity_uid(kind: str) -> str:
    normalized = str(kind or "").strip().lower()
    if normalized not in _ALLOWED_KINDS:
        raise ValueError(f"Unsupported entity uid kind: {kind!r}")
    return f"{normalized}_{new_ulid()}"


def ensure_entity_uid(value: str | None, kind: str) -> str:
    text = str(value or "").strip()
    return text if text else new_entity_uid(kind)
