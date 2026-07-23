"""Process-cold release benchmark for representative saved projects.

Run the parent on Windows. Each sample is executed in a fresh Python process;
the source project is opened read-only and any benchmark save goes to a
temporary directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _timer(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, (time.perf_counter() - started) * 1000.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _child(project_path: Path) -> dict[str, object]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import psutil
    from PySide6.QtWidgets import QApplication

    from app.application.proof_workspace import (
        build_proof_workspace_patch,
        build_proof_workspace_view,
    )
    from app.services.project_file_service import ProjectFileService
    from app.services.proof_session_service import ProofSessionService
    from app.ui.proof.formula_renderer import formula_preview_service
    from app.ui.proof.h_proof import HProofPanel
    from app.ui.proof.v_proof import VProofPanel

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.01):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    app = QApplication.instance() or QApplication([])
    service = ProjectFileService()
    timings: dict[str, float] = {}
    try:
        bound, timings["project_open_ms"] = _timer(lambda: service.open_project(project_path))
        workspace, timings["proof_projection_ms"] = _timer(
            lambda: build_proof_workspace_view(bound.session)
        )
        hpanel = HProofPanel()
        _, timings["hproof_publish_ms"] = _timer(lambda: hpanel.set_workspace(workspace))
        vpanel = VProofPanel()
        _, timings["vproof_publish_ms"] = _timer(lambda: vpanel.set_workspace(workspace))
        _, timings["formula_enqueue_ms"] = _timer(
            lambda: formula_preview_service().request(r"\\frac{x+1}{y-1}")
        )

        states = bound.session.proof_repository.all_states()
        if states and states[0].text_units:
            state = states[0]
            unit = state.text_units[0]
            proof_service = ProofSessionService(bound.session)

            def edit_and_project():
                result = proof_service.replace_text(
                    state.uid,
                    unit.uid,
                    unit.text + " ",
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    expected_unit_revision=unit.revision,
                    expected_unit_fingerprint=unit.fingerprint,
                )
                return build_proof_workspace_patch(
                    bound.session,
                    {state.uid: result.changed_text_unit_uids},
                )

            _patch, timings["proof_edit_and_patch_ms"] = _timer(edit_and_project)
        with tempfile.TemporaryDirectory(prefix="ocr-release-benchmark-") as directory:
            target = Path(directory) / project_path.name
            _, timings["save_as_ms"] = _timer(
                lambda: service.save_as(bound.session, target)
            )
        app.processEvents()
        return {
            "project": str(project_path.resolve()),
            "page_count": len(bound.session.page_repository.all()),
            "proof_state_count": len(bound.session.proof_repository.all_states()),
            "timings_ms": timings,
            "peak_rss_mib": peak_rss / (1024 * 1024),
        }
    finally:
        stop.set()
        sampler.join(timeout=1.0)


def _parent(projects: list[Path], runs: int, output: Path) -> None:
    samples: list[dict[str, object]] = []
    for project in projects:
        for run in range(runs):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(project)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            sample = json.loads(completed.stdout)
            sample["run"] = run + 1
            samples.append(sample)
    metrics = sorted({
        name
        for sample in samples
        for name in sample["timings_ms"]
    })
    summary = {
        name: {
            "p50_ms": statistics.median(values),
            "p95_ms": _percentile(values, 0.95),
        }
        for name in metrics
        if (values := [
            float(sample["timings_ms"][name])
            for sample in samples
            if name in sample["timings_ms"]
        ])
    }
    report = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "runs_per_project": runs,
        "summary": summary,
        "peak_rss_mib": {
            "p50": statistics.median(float(item["peak_rss_mib"]) for item in samples),
            "p95": _percentile([float(item["peak_rss_mib"]) for item in samples], 0.95),
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", action="append", type=Path, default=[])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark-release.json"))
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(_child(args.child), ensure_ascii=False))
        return
    if not args.project:
        parser.error("at least one --project is required")
    if args.runs < 1:
        parser.error("--runs must be positive")
    missing = [str(path) for path in args.project if not path.is_file()]
    if missing:
        parser.error("project does not exist: " + ", ".join(missing))
    _parent(args.project, args.runs, args.output)


if __name__ == "__main__":
    main()
