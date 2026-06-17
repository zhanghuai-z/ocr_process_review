"""Content-addressed cache for Hanwang native probe JSON outputs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA = "hanwang-native-probe-cache-v1"


def cache_enabled() -> bool:
    value = os.environ.get("HANWANG_NATIVE_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def cache_dir() -> Path:
    configured = os.environ.get("HANWANG_NATIVE_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path.cwd() / ".cache" / "hanwang_native"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _fingerprint_cached(path_text: str, size: int, mtime_ns: int) -> str:
    path = Path(path_text)
    return _sha256_file(path)


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return _fingerprint_cached(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def native_fingerprint(paths: list[Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            result[path.name] = {
                "path": str(resolved),
                "sha256": file_fingerprint(resolved),
            }
    return result


def image_fingerprint(image_bgr: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(image_bgr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": digest.hexdigest(),
    }


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(kind: str, key: str) -> Path:
    safe_kind = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in kind)
    return cache_dir() / safe_kind / f"{key}.json"


def read_json(kind: str, key: str) -> dict[str, Any] | None:
    if not cache_enabled():
        return None
    path = cache_path(kind, key)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return None


def write_json(kind: str, key: str, value: dict[str, Any]) -> None:
    if not cache_enabled():
        return
    tmp: Path | None = None
    try:
        path = cache_path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(path)
    except Exception:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


__all__ = [
    "CACHE_SCHEMA",
    "cache_enabled",
    "cache_key",
    "cache_path",
    "file_fingerprint",
    "image_fingerprint",
    "native_fingerprint",
    "read_json",
    "write_json",
]
