from __future__ import annotations

from pathlib import Path

import numpy as np


def test_linecut_segimg_cache_reuses_identical_probe_result(tmp_path, monkeypatch):
    from app.engines.hanwang import native_bridge

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "linecut_segimg_probe.exe").write_bytes(b"segimg-stub")
    (bin_dir / "linecut.dll").write_bytes(b"linecut-stub")
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"run": 0}

    def fake_get_bin_dir():
        return bin_dir

    def fake_save_temp_image(image_bgr, work_dir):
        path = work_dir / f"_fake_{calls['run']}.png"
        path.write_bytes(b"png")
        return path

    def fake_run_exe(exe, args, *, cwd, timeout):
        calls["run"] += 1
        (Path(cwd) / args[1]).write_text(
            '{"lines":[{"groups":[{"bbox":{"left":1,"top":2,"right":3,"bottom":4}}]}]}',
            encoding="utf-8",
        )
        return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(native_bridge, "get_hanwang_bin_dir", fake_get_bin_dir)
    monkeypatch.setattr(native_bridge, "_save_temp_image", fake_save_temp_image)
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    first = native_bridge.run_linecut_segimg(image, recblocks_xyxy=[(1, 2, 9, 8)])
    second = native_bridge.run_linecut_segimg(image, recblocks_xyxy=[(1, 2, 9, 8)])

    assert calls["run"] == 1
    assert second == first


def test_linecut_recog_cache_key_includes_recognition_params(tmp_path, monkeypatch):
    from app.engines.hanwang import native_bridge

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "linecut_recogimg_probe.exe").write_bytes(b"recog-stub")
    (bin_dir / "linecut.dll").write_bytes(b"linecut-stub")
    (bin_dir / "IntegratRcg.dll").write_bytes(b"charrcg-stub")
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"run": 0}

    def fake_get_bin_dir():
        return bin_dir

    def fake_save_temp_image(image_bgr, work_dir):
        path = work_dir / f"_fake_{calls['run']}.png"
        path.write_bytes(b"png")
        return path

    def fake_run_exe(exe, args, *, cwd, timeout):
        calls["run"] += 1
        (Path(cwd) / args[1]).write_text(
            f'{{"lines":[],"call":{calls["run"]}}}',
            encoding="utf-8",
        )
        return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(native_bridge, "get_hanwang_bin_dir", fake_get_bin_dir)
    monkeypatch.setattr(native_bridge, "_save_temp_image", fake_save_temp_image)
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    first = native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)
    second = native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)
    third = native_bridge.run_linecut_recog(image, mode=75, postprocess=1, split_mode=0)

    assert calls["run"] == 2
    assert second == first
    assert third["call"] == 2


def test_eng20_recogline_cache_reuses_identical_probe_result(tmp_path, monkeypatch):
    from app.engines.hanwang import native_bridge

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in (
        "eng20_probe.exe",
        "Eng20.dll",
        "CutEng.dll",
        "EngDigital.dll",
        "HWEng20.db",
        "ENWList.db",
    ):
        (bin_dir / name).write_bytes(f"{name}-stub".encode("ascii"))
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"run": 0}

    def fake_get_bin_dir():
        return bin_dir

    def fake_save_temp_image(image_bgr, work_dir):
        path = work_dir / f"_fake_{calls['run']}.png"
        path.write_bytes(b"png")
        return path

    def fake_run_exe(exe, args, *, cwd, timeout):
        calls["run"] += 1
        (Path(cwd) / args[1]).write_text(
            f'{{"lines":[],"call":{calls["run"]}}}',
            encoding="utf-8",
        )
        return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(native_bridge, "get_hanwang_bin_dir", fake_get_bin_dir)
    monkeypatch.setattr(native_bridge, "_save_temp_image", fake_save_temp_image)
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    first = native_bridge.run_eng20_recogline(image)
    second = native_bridge.run_eng20_recogline(image)
    third = native_bridge.run_eng20_recogline(np.ones((10, 12, 3), dtype=np.uint8))

    assert calls["run"] == 2
    assert second == first
    assert third["call"] == 2


def test_native_cache_write_json_is_fail_open_when_cache_dir_is_unwritable(tmp_path, monkeypatch):
    from app.engines.hanwang import native_cache

    blocked = tmp_path / "not-a-dir"
    blocked.write_text("blocked", encoding="utf-8")
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(blocked))

    native_cache.write_json("linecut_segimg", "cache-key", {"ok": True})
