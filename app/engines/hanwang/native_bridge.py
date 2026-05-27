"""subprocess 桥接层：把 64 位 Python 调用翻译成 32 位 native probe.exe 调用。

设计要点：
- probe.exe 必须**在自己所在目录**运行（CWD = bin_dir），否则 DLL 旁加载找不到
- 图像路径以 **本地短路径** 传入（用 tempfile，避免中文/空格），Bitmap 解码才稳
- 输出 JSON 立刻读回内存，临时文件由调用方在 with 块外清理
- 所有错误（exit != 0、JSON 解析失败、AccessViolation 文本）统一抛 HanwangNativeError
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.engines.hanwang.paths import get_hanwang_bin_dir

logger = get_logger(__name__)


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


def _run_exe(exe: Path, args: list[str], *, cwd: Path, timeout: float) -> _ProbeRun:
    """同步运行 probe.exe，返回标准输出/错误。"""
    cmd = [str(exe), *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
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
    img_path = _save_temp_image(image_bgr, bin_dir)
    out_path = bin_dir / f"{img_path.stem}.docseg.json"
    try:
        run = _run_exe(
            exe,
            [img_path.name, out_path.name],
            cwd=bin_dir,
            timeout=timeout,
        )
        if run.returncode != 0 or not out_path.is_file():
            raise HanwangNativeError(
                f"docseg_probe 失败 (rc={run.returncode}): {run.stderr.strip() or run.stdout.strip()}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        for p in (img_path, out_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


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
    img_path = _save_temp_image(image_bgr, bin_dir)
    out_path = bin_dir / f"{img_path.stem}.segimg.json"
    rb_path = bin_dir / f"{img_path.stem}.segrb.tsv"
    h, w = image_bgr.shape[:2]
    if not recblocks_xyxy:
        recblocks_xyxy = [(0, 0, w, h)]
    rb_path.write_text(
        "\n".join(f"{l}\t{t}\t{r}\t{b}" for (l, t, r, b) in recblocks_xyxy) + "\n",
        encoding="utf-8",
    )
    try:
        run = _run_exe(
            exe,
            [img_path.name, out_path.name, rb_path.name],
            cwd=bin_dir,
            timeout=timeout,
        )
        if run.returncode != 0 or not out_path.is_file():
            raise HanwangNativeError(
                f"linecut_segimg_probe 失败 (rc={run.returncode}): "
                f"{run.stderr.strip() or run.stdout.strip()}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        for p in (img_path, out_path, rb_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


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
    img_path = _save_temp_image(image_bgr, bin_dir)
    out_path = bin_dir / f"{img_path.stem}.linecut.json"
    rb_path = bin_dir / f"{img_path.stem}.rb.tsv"
    h, w = image_bgr.shape[:2]
    if recblocks_xyxy:
        rb_lines = [
            f"{int(l)}\t{int(t)}\t{int(r)}\t{int(b)}"
            for (l, t, r, b) in recblocks_xyxy
        ]
        rb_path.write_text("\n".join(rb_lines) + "\n", encoding="utf-8")
    elif recblock_xyxy is None:
        rb_l, rb_t, rb_r, rb_b = 0, 0, w, h
        rb_path.write_text(f"{rb_l}\t{rb_t}\t{rb_r}\t{rb_b}\n", encoding="utf-8")
    else:
        rb_l, rb_t, rb_r, rb_b = recblock_xyxy
        rb_path.write_text(f"{rb_l}\t{rb_t}\t{rb_r}\t{rb_b}\n", encoding="utf-8")

    args = [
        img_path.name,
        out_path.name,
        rb_path.name,
        str(mode),
        str(postprocess),
        str(split_mode),
        "via-seg",
    ]
    if with_charrcg:
        args.append("with-charrcg")

    try:
        run = _run_exe(exe, args, cwd=bin_dir, timeout=timeout)
        if run.returncode != 0 or not out_path.is_file():
            raise HanwangNativeError(
                f"linecut_recogimg_probe 失败 (rc={run.returncode}): "
                f"{run.stderr.strip() or run.stdout.strip()}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        for p in (img_path, out_path, rb_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
