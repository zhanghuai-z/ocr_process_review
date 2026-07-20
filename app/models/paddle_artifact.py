"""Append-only normalized vendor artifacts from Paddle layout analysis."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json


@dataclass(frozen=True, slots=True)
class PaddleArtifact:
    """One immutable Paddle response associated with an imported page image."""

    project_uid: str
    uid: str
    page_uid: str
    source_engine: str
    source_run_id: str
    image_hash: str
    payload_json: str
    artifact_path: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("project_uid", "uid", "page_uid", "source_engine", "source_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        for name in ("image_hash", "payload_json", "artifact_path"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be str")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json must contain valid JSON") from exc
        if not isinstance(payload, (dict, list)):
            raise ValueError("payload_json root must be an object or array")
        canonical = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "fingerprint"
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "fingerprint", hashlib.sha256(encoded).hexdigest())


__all__ = ["PaddleArtifact"]
