from __future__ import annotations

import json
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
    monkeypatch.setattr(native_bridge, "_native_path_arg", lambda path: str(path))
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
    monkeypatch.setattr(native_bridge, "_native_path_arg", lambda path: str(path))
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    first = native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)
    second = native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)
    third = native_bridge.run_linecut_recog(image, mode=75, postprocess=1, split_mode=0)

    assert calls["run"] == 2
    assert second == first
    assert third["call"] == 2


def test_linecut_recog_batch_list_reuses_cache_and_preserves_order(tmp_path, monkeypatch):
    from app.engines.hanwang import native_bridge

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in (
        "linecut_recog_batch_probe.exe",
        "linecut_recogimg_probe.exe",
        "linecut.dll",
        "IntegratRcg.dll",
    ):
        (bin_dir / name).write_bytes(f"{name}-stub".encode("ascii"))
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"run": 0, "save": 0}

    def fake_get_bin_dir():
        return bin_dir

    def fake_save_temp_image(image_bgr, work_dir):
        calls["save"] += 1
        path = work_dir / f"_fake_{calls['save']}.png"
        path.write_bytes(b"png")
        return path

    def fake_run_exe(exe, args, *, cwd, timeout):
        calls["run"] += 1
        assert Path(exe).name == "linecut_recog_batch_probe.exe"
        assert args[0] == "--batch-list"
        tasks_path = Path(args[1])
        summary_path = Path(args[2])
        tasks = []
        for idx, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines()):
            image_path, output_path = line.split("\t")
            Path(output_path).write_text(
                json.dumps({"lines": [], "run": calls["run"], "task": idx, "image": image_path}),
                encoding="utf-8",
            )
            tasks.append({
                "index": idx,
                "image": image_path,
                "output": output_path,
                "ok": True,
                "lines": 0,
                "chars": 0,
                "error": "",
            })
        summary_path.write_text(
            json.dumps({"schema": "linecut_recog_batch_probe.v1", "tasks": tasks}),
            encoding="utf-8",
        )
        return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(native_bridge, "get_hanwang_bin_dir", fake_get_bin_dir)
    monkeypatch.setattr(native_bridge, "_save_temp_image", fake_save_temp_image)
    monkeypatch.setattr(native_bridge, "_native_path_arg", lambda path: str(path))
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    images = [
        np.zeros((10, 12, 3), dtype=np.uint8),
        np.ones((8, 9, 3), dtype=np.uint8),
    ]
    first = native_bridge.run_linecut_recog_batch_list(images, mode=71, postprocess=1, split_mode=0)
    second = native_bridge.run_linecut_recog_batch_list(images, mode=71, postprocess=1, split_mode=0)

    assert calls["run"] == 1
    assert [item["task"] for item in first] == [0, 1]
    assert second == first


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
    monkeypatch.setattr(native_bridge, "_native_path_arg", lambda path: str(path))
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


def test_native_cache_runtime_dir_and_clear_cache(tmp_path, monkeypatch):
    from app.engines.hanwang import native_cache

    monkeypatch.delenv("HANWANG_NATIVE_CACHE_DIR", raising=False)
    runtime_dir = tmp_path / "project" / ".cache" / "hanwang_native"
    try:
        native_cache.set_runtime_cache_dir(runtime_dir)

        assert native_cache.cache_dir() == runtime_dir
        native_cache.write_json("linecut_segimg", "cache-key", {"ok": True})
        assert native_cache.read_json("linecut_segimg", "cache-key") == {"ok": True}

        cleared = native_cache.clear_cache()

        assert cleared == runtime_dir
        assert not runtime_dir.exists()
        assert native_cache.read_json("linecut_segimg", "cache-key") is None
    finally:
        native_cache.reset_runtime_cache_dir()


def test_native_cache_env_dir_overrides_runtime_dir(tmp_path, monkeypatch):
    from app.engines.hanwang import native_cache

    runtime_dir = tmp_path / "runtime-cache"
    env_dir = tmp_path / "env-cache"
    try:
        native_cache.set_runtime_cache_dir(runtime_dir)
        monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(env_dir))

        assert native_cache.cache_dir() == env_dir
    finally:
        native_cache.reset_runtime_cache_dir()


def test_native_bridge_timing_log_records_cache_hit_and_miss(tmp_path, monkeypatch):
    from app.engines.hanwang import native_bridge

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "linecut_recogimg_probe.exe").write_bytes(b"recog-stub")
    (bin_dir / "linecut.dll").write_bytes(b"linecut-stub")
    (bin_dir / "IntegratRcg.dll").write_bytes(b"charrcg-stub")
    log_path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("HANWANG_NATIVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HANWANG_NATIVE_TIMING_LOG", str(log_path))
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
            '{"lines":[]}',
            encoding="utf-8",
        )
        return native_bridge._ProbeRun(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(native_bridge, "get_hanwang_bin_dir", fake_get_bin_dir)
    monkeypatch.setattr(native_bridge, "_save_temp_image", fake_save_temp_image)
    monkeypatch.setattr(native_bridge, "_native_path_arg", lambda path: str(path))
    monkeypatch.setattr(native_bridge, "_run_exe", fake_run_exe)

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)
    native_bridge.run_linecut_recog(image, mode=71, postprocess=1, split_mode=0)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert calls["run"] == 1
    assert len(records) == 2
    assert records[0]["probe"] == "linecut_recog"
    assert records[0]["cache_hit"] is False
    assert records[0]["ok"] is True
    assert "subprocess_seconds" in records[0]
    assert "temp_image_write_seconds" in records[0]
    assert records[1]["cache_hit"] is True
    assert records[1]["ok"] is True
    assert "subprocess_seconds" not in records[1]
