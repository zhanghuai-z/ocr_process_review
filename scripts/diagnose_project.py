"""Run read-only diagnostics against an OCR Process project file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import quality_probe as qp
from app.core.project_store import ProjectStore
from app.services.project_diagnostics import diagnose_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose proof/OCR text and geometry drift in a .ocrproj file.",
    )
    parser.add_argument("project", help="Path to .ocrproj")
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--qprobe", default="", help="Optional .qprobe.json sidecar path")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--fail-on",
        choices=("none", "warning", "error"),
        default="error",
        help="Exit non-zero when diagnostics reach this severity.",
    )
    args = parser.parse_args(argv)

    with ProjectStore(args.project) as store:
        project = store.load_project(args.project_id)

    probe_store = None
    sidecar = args.qprobe or qp.sidecar_path_for_project(args.project)
    if sidecar:
        probe_store = qp.load_store_from_path(sidecar)

    report = diagnose_project(project, probe_store=probe_store)
    if args.format == "markdown":
        print(report.to_markdown())
    else:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.fail_on == "error" and report.error_count:
        return 2
    if args.fail_on == "warning" and report.issue_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

