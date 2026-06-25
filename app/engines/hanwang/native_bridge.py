"""subprocess 桥接层：把 64 位 Python 调用翻译成 32 位 native probe.exe 调用。

设计要点：
- probe.exe 必须**在自己所在目录**运行（CWD = bin_dir），否则 DLL 旁加载找不到
- 图像路径以 **本地短路径** 传入（用 tempfile，避免中文/空格），Bitmap 解码才稳
- 输出 JSON 立刻读回内存，临时文件由调用方在 with 块外清理
- 所有错误（exit != 0、JSON 解析失败、AccessViolation 文本）统一抛 HanwangNativeError
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.engines.hanwang.paths import get_hanwang_bin_dir
from app.engines.hanwang import native_cache

logger = get_logger(__name__)
_TIMING_LOG_LOCK = threading.Lock()


# 简体中文识别参数（与 hanwang_native_workflow.py 的 AUTOREC_CHINESE_LANGUAGE_MODES["简体"] 保持一致）
RECOG_MODE_SIMPLIFIED: int = 0x47  # 71
RECOG_MODE_TRADITIONAL: int = 0x4B  # 75
RECOG_POSTPROCESS_DEFAULT: int = 1


class HanwangNativeError(RuntimeError):
    """probe.exe 调用失败的统一异常。"""


@dataclass
class _ProbeRun:
    stdout: str
    stderr: str
    returncode: int


def _timing_enabled() -> bool:
    value = os.environ.get("HANWANG_NATIVE_TIMING", "").strip().lower()
    return bool(os.environ.get("HANWANG_NATIVE_TIMING_LOG", "").strip()) or value in {
        "1",
        "true",
        "yes",
        "on",
        "enable",
        "enabled",
    }


def _timing_log_path() -> Path:
    configured = os.environ.get("HANWANG_NATIVE_TIMING_LOG", "").strip()
    if configured:
        return Path(configured)
    return native_cache.cache_dir().parent / "hanwang_native_timing.jsonl"


def _stage_seconds(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _record_stage(timing: dict[str, Any] | None, name: str, started: float) -> None:
    if timing is not None:
        timing[name] = _stage_seconds(started)


def _emit_timing(timing: dict[str, Any] | None) -> None:
    if timing is None:
        return
    timing.setdefault("schema", "hanwang-native-timing-v1")
    timing.setdefault("timestamp_unix", round(time.time(), 6))
    try:
        path = _timing_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(timing, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with _TIMING_LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(encoded)
                fh.write("\n")
    except Exception:
        logger.debug("Failed to write Hanwang native timing log", exc_info=True)


def _run_exe(exe: Path, args: list[str], *, cwd: Path, timeout: float) -> _ProbeRun:
    """同步运行 probe.exe，返回标准输出/错误。"""
    cmd = [str(exe), *args]
    env = os.environ.copy()
    env["PATH"] = str(cwd) + os.pathsep + env.get("PATH", "")
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            creationflags=creationflags,
            startupinfo=startupinfo,
            capture_output=True,
            timeout=timeout,
            # probe 内部用 GBK 输出错误信息（中文乱码可读性差，但不影响 returncode 判断）
            text=False,
        )
    except subprocess.TimeoutExpired as e:
        raise HanwangNativeError(f"probe 超时（{timeout}s）：{exe.name}") from e
    except FileNotFoundError as e:
        raise HanwangNativeError(f"probe 不存在：{exe}") from e

    stderr = proc.stderr.decode("gbk", errors="replace") if proc.stderr else ""
    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    return _ProbeRun(stdout=stdout, stderr=stderr, returncode=proc.returncode)


def _is_wsl() -> bool:
    if os.name == "nt":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _native_path_arg(path: Path) -> str:
    """Return a path string readable by Windows probe.exe under native Windows or WSL."""
    resolved = path.resolve()
    if os.name == "nt":
        return str(resolved)
    text = resolved.as_posix()
    if text.startswith("/mnt/") and len(text) > 6 and text[5].isalpha() and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    distro = os.environ.get("WSL_DISTRO_NAME", "").strip()
    if distro:
        return f"\\\\wsl.localhost\\{distro}" + text.replace("/", "\\")
    return str(resolved)


def _probe_work_base() -> Path:
    configured = os.environ.get("HANWANG_NATIVE_WORK_DIR", "").strip()
    if configured:
        return Path(configured)
    if _is_wsl():
        cwd = Path.cwd().resolve()
        if cwd.as_posix().startswith("/mnt/"):
            return cwd / ".cache" / "hanwang_native_work"
    return Path(tempfile.gettempdir()) / "ocr_process_hanwang_native"


@contextmanager
def _probe_work_dir() -> Iterator[Path]:
    """Create a writable per-call work dir for probe input/output files.

    PyInstaller puts bundled data under ``_internal`` / ``_MEIPASS``. Treat that
    location as read-only: it is for native assets only, not temporary OCR files.
    """
    try:
        base = _probe_work_base()
        base.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.TemporaryDirectory(prefix="probe_", dir=str(base))
    except OSError:
        tmp_dir = tempfile.TemporaryDirectory(prefix="ocr_hw_")
    with tmp_dir as tmp:
        yield Path(tmp)


def _save_temp_image(image_bgr: np.ndarray, work_dir: Path) -> Path:
    """把 BGR ndarray 落成 PNG 到工作目录，返回纯文件名（同目录运行）。"""
    import cv2

    name = f"_hw_{uuid.uuid4().hex[:12]}.png"
    out = work_dir / name
    ok = cv2.imwrite(str(out), image_bgr)
    if not ok:
        raise HanwangNativeError(f"无法写出临时 PNG: {out}")
    return out


def run_docseg(image_bgr: np.ndarray, *, timeout: float = 60.0) -> dict:
    """跑 doc_seg：把整页图像切成版面区域。

    返回原始 JSON dict，含 keys: width / height / areas[].
    """
    bin_dir = get_hanwang_bin_dir()
    exe = bin_dir / "docseg_probe.exe"
    with _probe_work_dir() as work_dir:
        img_path = _save_temp_image(image_bgr, work_dir)
        out_path = work_dir / f"{img_path.stem}.docseg.json"
        run = _run_exe(
            exe,
            [_native_path_arg(img_path), _native_path_arg(out_path)],
            cwd=bin_dir,
            timeout=timeout,
        )
        if run.returncode != 0 or not out_path.is_file():
            raise HanwangNativeError(
                f"docseg_probe 失败 (rc={run.returncode}): {run.stderr.strip() or run.stdout.strip()}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


def run_linecut_segimg(
    image_bgr: np.ndarray,
    *,
    recblocks_xyxy: Optional[list[tuple[int, int, int, int]]] = None,
    timeout: float = 60.0,
) -> dict:
    """跑 linecut.SegImg：把图像（含多行）切成 lines/groups/chars。

    Args:
        image_bgr: BGR 图像（可以是整页，也可以是 area crop）
        recblocks_xyxy: 识别区域链 (l,t,r,b) 列表，None = 整图作为一个 recblock

    返回 dict，含 keys: width/height/lines[]，line.groups[].bbox 为 image_bgr 局部坐标。
    """
    bin_dir = get_hanwang_bin_dir()
    exe = bin_dir / "linecut_segimg_probe.exe"
    if not exe.is_file():
        raise HanwangNativeError(f"linecut_segimg_probe.exe 缺失：{exe}")
    h, w = image_bgr.shape[:2]
    if not recblocks_xyxy:
        recblocks_xyxy = [(0, 0, w, h)]
    recblocks_xyxy = [
        (int(l), int(t), int(r), int(b))
        for (l, t, r, b) in recblocks_xyxy
    ]
    timing: dict[str, Any] | None = None
    total_started = time.perf_counter()
    if _timing_enabled():
        timing = {
            "probe": "linecut_segimg",
            "exe": exe.name,
            "image_shape": [int(item) for item in image_bgr.shape],
            "image_pixels": int(w * h),
            "recblock_count": len(recblocks_xyxy),
            "cache_hit": False,
            "ok": False,
        }
    try:
        started = time.perf_counter()
        image_fingerprint = native_cache.image_fingerprint(image_bgr)
        _record_stage(timing, "image_fingerprint_seconds", started)
        started = time.perf_counter()
        native_fingerprint = native_cache.native_fingerprint([exe, bin_dir / "linecut.dll"])
        _record_stage(timing, "native_fingerprint_seconds", started)
        started = time.perf_counter()
        key_payload = {
            "schema": native_cache.CACHE_SCHEMA,
            "probe": "linecut_segimg",
            "image": image_fingerprint,
            "recblocks_xyxy": [list(item) for item in recblocks_xyxy],
            "native": native_fingerprint,
        }
        cache_key = native_cache.cache_key(key_payload)
        if timing is not None:
            timing["cache_key"] = cache_key[:16]
        _record_stage(timing, "cache_key_seconds", started)
        started = time.perf_counter()
        cached = native_cache.read_json("linecut_segimg", cache_key)
        _record_stage(timing, "cache_read_seconds", started)
        if cached is not None:
            if timing is not None:
                timing["cache_hit"] = True
                timing["ok"] = True
            return cached

        started = time.perf_counter()
        with _probe_work_dir() as work_dir:
            _record_stage(timing, "workdir_seconds", started)
            started = time.perf_counter()
            img_path = _save_temp_image(image_bgr, work_dir)
            _record_stage(timing, "temp_image_write_seconds", started)
            out_path = work_dir / f"{img_path.stem}.segimg.json"
            rb_path = work_dir / f"{img_path.stem}.segrb.tsv"
            started = time.perf_counter()
            rb_path.write_text(
                "\n".join(f"{l}\t{t}\t{r}\t{b}" for (l, t, r, b) in recblocks_xyxy) + "\n",
                encoding="utf-8",
            )
            _record_stage(timing, "recblock_write_seconds", started)
            started = time.perf_counter()
            run = _run_exe(
                exe,
                [_native_path_arg(img_path), _native_path_arg(out_path), _native_path_arg(rb_path)],
                cwd=bin_dir,
                timeout=timeout,
            )
            _record_stage(timing, "subprocess_seconds", started)
            if run.returncode != 0 or not out_path.is_file():
                raise HanwangNativeError(
                    f"linecut_segimg_probe 失败 (rc={run.returncode}): "
                    f"{run.stderr.strip() or run.stdout.strip()}"
                )
            started = time.perf_counter()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            _record_stage(timing, "output_json_read_seconds", started)
            started = time.perf_counter()
            native_cache.write_json("linecut_segimg", cache_key, payload)
            _record_stage(timing, "cache_write_seconds", started)
            if timing is not None:
                timing["ok"] = True
            return payload
    except Exception as exc:
        if timing is not None:
            timing["error"] = str(exc)
        raise
    finally:
        if timing is not None:
            timing["total_seconds"] = _stage_seconds(total_started)
        _emit_timing(timing)


def run_linecut_recog(
    image_bgr: np.ndarray,
    *,
    recblock_xyxy: Optional[tuple[int, int, int, int]] = None,
    recblocks_xyxy: Optional[list[tuple[int, int, int, int]]] = None,
    with_charrcg: bool = True,
    mode: int = RECOG_MODE_SIMPLIFIED,
    postprocess: int = RECOG_POSTPROCESS_DEFAULT,
    split_mode: int = 0,
    timeout: float = 120.0,
) -> dict:
    """跑 linecut.SegImg + Recog (+ 可选 IntegratRcg.CharRcg)。

    Args:
        image_bgr: BGR 图像（应当是单个 area 的 crop）
        recblock_xyxy: 单个识别区域 (l, t, r, b)，None = 全图
        recblocks_xyxy: 多个识别区域；用于一次 probe 处理多个 group，避免 per-group 冷启动
        with_charrcg: 是否启用字符级 CharRcg（候选 top-10 + 字框）

    返回原始 JSON dict（schema 见 linecut_recogimg_probe.cs.BuildJson）。
    """
    if recblock_xyxy is not None and recblocks_xyxy is not None:
        raise ValueError("recblock_xyxy and recblocks_xyxy are mutually exclusive")
    bin_dir = get_hanwang_bin_dir()
    exe = bin_dir / "linecut_recogimg_probe.exe"
    h, w = image_bgr.shape[:2]
    if recblocks_xyxy:
        normalized_recblocks = [
            (int(l), int(t), int(r), int(b))
            for (l, t, r, b) in recblocks_xyxy
        ]
    elif recblock_xyxy is None:
        normalized_recblocks = [(0, 0, w, h)]
    else:
        rb_l, rb_t, rb_r, rb_b = recblock_xyxy
        normalized_recblocks = [(int(rb_l), int(rb_t), int(rb_r), int(rb_b))]

    args = [
        "",  # filled after temporary files are created
        "",
        "",
        str(mode),
        str(postprocess),
        str(split_mode),
        "via-seg",
    ]
    if with_charrcg:
        args.append("with-charrcg")
    timing: dict[str, Any] | None = None
    total_started = time.perf_counter()
    if _timing_enabled():
        timing = {
            "probe": "linecut_recog",
            "exe": exe.name,
            "image_shape": [int(item) for item in image_bgr.shape],
            "image_pixels": int(w * h),
            "recblock_count": len(normalized_recblocks),
            "with_charrcg": bool(with_charrcg),
            "mode": int(mode),
            "postprocess": int(postprocess),
            "split_mode": int(split_mode),
            "cache_hit": False,
            "ok": False,
        }
    try:
        started = time.perf_counter()
        image_fingerprint = native_cache.image_fingerprint(image_bgr)
        _record_stage(timing, "image_fingerprint_seconds", started)
        started = time.perf_counter()
        native_fingerprint = native_cache.native_fingerprint([
            exe,
            bin_dir / "linecut.dll",
            bin_dir / "IntegratRcg.dll",
        ])
        _record_stage(timing, "native_fingerprint_seconds", started)
        started = time.perf_counter()
        key_payload = {
            "schema": native_cache.CACHE_SCHEMA,
            "probe": "linecut_recog",
            "image": image_fingerprint,
            "recblocks_xyxy": [list(item) for item in normalized_recblocks],
            "with_charrcg": bool(with_charrcg),
            "mode": int(mode),
            "postprocess": int(postprocess),
            "split_mode": int(split_mode),
            "via_seg": True,
            "native": native_fingerprint,
        }
        cache_key = native_cache.cache_key(key_payload)
        if timing is not None:
            timing["cache_key"] = cache_key[:16]
        _record_stage(timing, "cache_key_seconds", started)
        started = time.perf_counter()
        cached = native_cache.read_json("linecut_recog", cache_key)
        _record_stage(timing, "cache_read_seconds", started)
        if cached is not None:
            if timing is not None:
                timing["cache_hit"] = True
                timing["ok"] = True
            return cached

        started = time.perf_counter()
        with _probe_work_dir() as work_dir:
            _record_stage(timing, "workdir_seconds", started)
            started = time.perf_counter()
            img_path = _save_temp_image(image_bgr, work_dir)
            _record_stage(timing, "temp_image_write_seconds", started)
            out_path = work_dir / f"{img_path.stem}.linecut.json"
            rb_path = work_dir / f"{img_path.stem}.rb.tsv"
            started = time.perf_counter()
            rb_path.write_text(
                "\n".join(f"{l}\t{t}\t{r}\t{b}" for (l, t, r, b) in normalized_recblocks) + "\n",
                encoding="utf-8",
            )
            _record_stage(timing, "recblock_write_seconds", started)
            args[0] = _native_path_arg(img_path)
            args[1] = _native_path_arg(out_path)
            args[2] = _native_path_arg(rb_path)

            started = time.perf_counter()
            run = _run_exe(exe, args, cwd=bin_dir, timeout=timeout)
            _record_stage(timing, "subprocess_seconds", started)
            if run.returncode != 0 or not out_path.is_file():
                raise HanwangNativeError(
                    f"linecut_recogimg_probe 失败 (rc={run.returncode}): "
                    f"{run.stderr.strip() or run.stdout.strip()}"
                )
            started = time.perf_counter()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            _record_stage(timing, "output_json_read_seconds", started)
            started = time.perf_counter()
            native_cache.write_json("linecut_recog", cache_key, payload)
            _record_stage(timing, "cache_write_seconds", started)
            if timing is not None:
                timing["ok"] = True
            return payload
    except Exception as exc:
        if timing is not None:
            timing["error"] = str(exc)
        raise
    finally:
        if timing is not None:
            timing["total_seconds"] = _stage_seconds(total_started)
        _emit_timing(timing)


def run_linecut_recog_batch_list(
    images_bgr: list[np.ndarray],
    *,
    with_charrcg: bool = True,
    mode: int = RECOG_MODE_SIMPLIFIED,
    postprocess: int = RECOG_POSTPROCESS_DEFAULT,
    split_mode: int = 0,
    timeout: float = 120.0,
) -> list[dict]:
    """Run linecut Recog for many independent crop images in one wrapper process.

    This intentionally does *not* pass multiple recblocks to one native Recog
    call. The wrapper process loops over safe single-recblock crop calls, so it
    reduces process startup/init overhead without using the unstable multi-RB
    native path.
    """
    if not images_bgr:
        return []
    bin_dir = get_hanwang_bin_dir()
    exe = bin_dir / "linecut_recog_batch_probe.exe"
    if not exe.is_file():
        raise HanwangNativeError(f"linecut_recog_batch_probe.exe 缺失：{exe}")

    source_exe = bin_dir / "linecut_recogimg_probe.exe"
    native_fingerprint = native_cache.native_fingerprint([
        exe,
        source_exe,
        bin_dir / "linecut.dll",
        bin_dir / "IntegratRcg.dll",
    ])
    results: list[dict | None] = [None] * len(images_bgr)
    misses: list[tuple[int, np.ndarray, str]] = []
    timing: dict[str, Any] | None = None
    total_started = time.perf_counter()
    if _timing_enabled():
        timing = {
            "probe": "linecut_recog_batch_list",
            "exe": exe.name,
            "task_count": len(images_bgr),
            "with_charrcg": bool(with_charrcg),
            "mode": int(mode),
            "postprocess": int(postprocess),
            "split_mode": int(split_mode),
            "cache_hits": 0,
            "cache_misses": 0,
            "ok": False,
        }

    try:
        started = time.perf_counter()
        for idx, image_bgr in enumerate(images_bgr):
            h, w = image_bgr.shape[:2]
            image_fingerprint = native_cache.image_fingerprint(image_bgr)
            key_payload = {
                "schema": native_cache.CACHE_SCHEMA,
                "probe": "linecut_recog_batch_list",
                "image": image_fingerprint,
                "recblocks_xyxy": [[0, 0, int(w), int(h)]],
                "with_charrcg": bool(with_charrcg),
                "mode": int(mode),
                "postprocess": int(postprocess),
                "split_mode": int(split_mode),
                "via_seg": True,
                "native": native_fingerprint,
            }
            cache_key = native_cache.cache_key(key_payload)
            cached = native_cache.read_json("linecut_recog_batch_list", cache_key)
            if cached is not None:
                results[idx] = cached
                if timing is not None:
                    timing["cache_hits"] += 1
            else:
                misses.append((idx, image_bgr, cache_key))
                if timing is not None:
                    timing["cache_misses"] += 1
        _record_stage(timing, "cache_prepare_seconds", started)

        if misses:
            started = time.perf_counter()
            with _probe_work_dir() as work_dir:
                _record_stage(timing, "workdir_seconds", started)
                tasks_path = work_dir / "linecut_recog_batch_tasks.tsv"
                summary_path = work_dir / "linecut_recog_batch_summary.json"
                task_records: list[tuple[int, str, Path]] = []
                task_lines: list[str] = []
                started = time.perf_counter()
                for task_idx, (result_idx, image_bgr, _cache_key) in enumerate(misses):
                    img_path = _save_temp_image(image_bgr, work_dir)
                    out_path = work_dir / f"{img_path.stem}.linecut.json"
                    task_records.append((result_idx, _cache_key, out_path))
                    task_lines.append(f"{_native_path_arg(img_path)}\t{_native_path_arg(out_path)}")
                tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
                _record_stage(timing, "task_file_write_seconds", started)

                args = [
                    "--batch-list",
                    _native_path_arg(tasks_path),
                    _native_path_arg(summary_path),
                    str(mode),
                    str(postprocess),
                    str(split_mode),
                    "via-seg",
                ]
                if with_charrcg:
                    args.append("with-charrcg")
                started = time.perf_counter()
                run = _run_exe(exe, args, cwd=bin_dir, timeout=timeout)
                _record_stage(timing, "subprocess_seconds", started)
                if run.returncode != 0 or not summary_path.is_file():
                    raise HanwangNativeError(
                        f"linecut_recog_batch_probe 失败 (rc={run.returncode}): "
                        f"{run.stderr.strip() or run.stdout.strip()}"
                    )
                started = time.perf_counter()
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                tasks = summary.get("tasks") if isinstance(summary, dict) else None
                if not isinstance(tasks, list) or len(tasks) != len(task_records):
                    raise HanwangNativeError(
                        f"linecut_recog_batch_probe 返回 task 数异常："
                        f"{0 if not isinstance(tasks, list) else len(tasks)}/{len(task_records)}"
                    )
                failed = [item for item in tasks if not isinstance(item, dict) or not item.get("ok")]
                if failed:
                    raise HanwangNativeError(f"linecut_recog_batch_probe task 失败：{len(failed)}")
                for result_idx, cache_key, out_path in task_records:
                    payload = json.loads(out_path.read_text(encoding="utf-8"))
                    results[result_idx] = payload
                    native_cache.write_json("linecut_recog_batch_list", cache_key, payload)
                _record_stage(timing, "output_json_read_seconds", started)

        finalized: list[dict] = []
        for item in results:
            if item is None:
                raise HanwangNativeError("linecut_recog_batch_probe 内部结果缺失")
            finalized.append(item)
        if timing is not None:
            timing["ok"] = True
        return finalized
    except Exception as exc:
        if timing is not None:
            timing["error"] = str(exc)
        raise
    finally:
        if timing is not None:
            timing["total_seconds"] = _stage_seconds(total_started)
        _emit_timing(timing)


def run_eng20_recogline(
    image_bgr: np.ndarray,
    *,
    timeout: float = 30.0,
) -> dict:
    """Run Eng20 on a single physical line crop.

    This is an English/Latin geometry probe. The caller must treat returned
    text as geometry evidence only; it is not authoritative OCR text for mixed
    Chinese/Latin lines.
    """
    bin_dir = get_hanwang_bin_dir()
    exe = bin_dir / "eng20_probe.exe"
    if not exe.is_file():
        raise HanwangNativeError(f"eng20_probe.exe 缺失：{exe}")

    h, w = image_bgr.shape[:2]
    timing: dict[str, Any] | None = None
    total_started = time.perf_counter()
    if _timing_enabled():
        timing = {
            "probe": "eng20_recogline",
            "exe": exe.name,
            "image_shape": [int(item) for item in image_bgr.shape],
            "image_pixels": int(w * h),
            "mode": "recogline_engstr",
            "packing": "packed",
            "direction": "tbrl",
            "cache_hit": False,
            "ok": False,
        }
    try:
        started = time.perf_counter()
        image_fingerprint = native_cache.image_fingerprint(image_bgr)
        _record_stage(timing, "image_fingerprint_seconds", started)
        started = time.perf_counter()
        native_fingerprint = native_cache.native_fingerprint([
            exe,
            bin_dir / "Eng20.dll",
            bin_dir / "CutEng.dll",
            bin_dir / "EngDigital.dll",
            bin_dir / "HWEng20.db",
            bin_dir / "ENWList.db",
        ])
        _record_stage(timing, "native_fingerprint_seconds", started)
        started = time.perf_counter()
        key_payload = {
            "schema": native_cache.CACHE_SCHEMA,
            "probe": "eng20_recogline",
            "image": image_fingerprint,
            "mode": "recogline_engstr",
            "packing": "packed",
            "direction": "tbrl",
            "native": native_fingerprint,
        }
        cache_key = native_cache.cache_key(key_payload)
        if timing is not None:
            timing["cache_key"] = cache_key[:16]
        _record_stage(timing, "cache_key_seconds", started)
        started = time.perf_counter()
        cached = native_cache.read_json("eng20_recogline", cache_key)
        _record_stage(timing, "cache_read_seconds", started)
        if cached is not None:
            if timing is not None:
                timing["cache_hit"] = True
                timing["ok"] = True
            return cached

        started = time.perf_counter()
        with _probe_work_dir() as work_dir:
            _record_stage(timing, "workdir_seconds", started)
            started = time.perf_counter()
            img_path = _save_temp_image(image_bgr, work_dir)
            _record_stage(timing, "temp_image_write_seconds", started)
            out_path = work_dir / f"{img_path.stem}.eng20.json"
            started = time.perf_counter()
            run = _run_exe(
                exe,
                [
                    _native_path_arg(img_path),
                    _native_path_arg(out_path),
                    "__missing_rb.tsv",
                    "recogline_engstr",
                    "packed",
                    "tbrl",
                ],
                cwd=bin_dir,
                timeout=timeout,
            )
            _record_stage(timing, "subprocess_seconds", started)
            if run.returncode != 0 or not out_path.is_file():
                raise HanwangNativeError(
                    f"eng20_probe 失败 (rc={run.returncode}): "
                    f"{run.stderr.strip() or run.stdout.strip()}"
                )
            started = time.perf_counter()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            _record_stage(timing, "output_json_read_seconds", started)
            started = time.perf_counter()
            native_cache.write_json("eng20_recogline", cache_key, payload)
            _record_stage(timing, "cache_write_seconds", started)
            if timing is not None:
                timing["ok"] = True
            return payload
    except Exception as exc:
        if timing is not None:
            timing["error"] = str(exc)
        raise
    finally:
        if timing is not None:
            timing["total_seconds"] = _stage_seconds(total_started)
        _emit_timing(timing)
