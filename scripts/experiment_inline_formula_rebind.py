#!/usr/bin/env python3
"""Build a Paddle pseudo page for ambiguous inline formula rebinding.

Default mode is offline: it writes the pseudo page and manifest only.
Use ``--call-paddle`` to submit the pseudo page to the configured PaddleOCR-VL
jobs API and write recognition/binding suggestions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.app_config import get_config  # noqa: E402
from app.core.project_store import ProjectStore  # noqa: E402
from app.core.paddle_v16_client import PaddleV16LayoutClient  # noqa: E402
from app.core.paddle_line_routing import ROUTE_SUBBLOCKS_FIELD, block_text  # noqa: E402
from app.core.api_profiles import resolve_api_endpoint_for_role  # noqa: E402
from app.core.api_profiles import FIXED_LAYOUT_PROFILE  # noqa: E402
from app.engines.hanwang.micro_recblock import _page_blocks_from_layout  # noqa: E402
from app.services.formula_crop_ocr_service import (  # noqa: E402
    build_formula_pseudo_page,
    formula_crop_bboxes_from_route_subblocks,
    recognize_formula_pseudo_page,
    recognitions_from_paddle_response,
)
from app.experimental.inline_formula_rebind import (  # noqa: E402
    suggest_formula_bindings,
)


def main() -> int:
    args = parse_args()
    project_path = Path(args.project)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_db_path = _snapshot_project_db(project_path, out_dir)

    store = ProjectStore(str(project_db_path))
    store.open()
    try:
        project_id = args.project_id if args.project_id is not None else _latest_project_id(store)
        project = store.load_project(project_id=project_id)
    finally:
        store.close()

    if not project.pages:
        raise SystemExit("project has no pages")
    page = _select_page(project.pages, args.page_index)
    image_path = _resolve_image_path(page.display_image_path, project_path.parent)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read page image: {page.display_image_path}")

    rows = _page_blocks_from_layout(page)
    parent = _select_parent_row(rows, args.parent_index)
    formula_bboxes = formula_crop_bboxes_from_route_subblocks(parent.get(ROUTE_SUBBLOCKS_FIELD, []))
    if not formula_bboxes:
        raise SystemExit("selected parent has no inline formula route subblocks")

    pseudo = build_formula_pseudo_page(image, formula_bboxes)
    pseudo_path = out_dir / "inline_formula_pseudo_page.png"
    cv2.imwrite(str(pseudo_path), pseudo.image_bgr)

    manifest = {
        "project": str(project_path),
        "project_snapshot": str(project_db_path),
        "project_id": project_id,
        "page_uid": page.uid,
        "page_number": page.page_number,
        "image_path": str(image_path),
        "parent_index": args.parent_index,
        "parent_label": parent.get("block_label") or parent.get("label"),
        "parent_bbox": parent.get("block_bbox") or parent.get("bbox"),
        "parent_text": block_text(parent),
        "crops": [asdict(crop) for crop in pseudo.crops],
        "pseudo_page": str(pseudo_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.response_json:
        response = json.loads(Path(args.response_json).read_text(encoding="utf-8"))
        _write_recognition_outputs(out_dir, pseudo, response, manifest["parent_text"])

    if args.call_paddle:
        cfg = get_config()
        url = resolve_api_endpoint_for_role(
            cfg.get("api_url", ""),
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        if not url:
            raise SystemExit("api_url is not configured")
        client = PaddleV16LayoutClient(
            jobs_url=url,
            token=str(cfg.get("api_token") or ""),
            request_timeout=max(10, int(cfg.get("api_timeout") or 30)),
            poll_timeout=max(120, int(cfg.get("api_timeout") or 30)),
            network_mode=str(cfg.get("paddle_api_network_mode") or "auto"),
            status_callback=lambda message: print(f"[paddle] {message}"),
        )
        recognitions, response = recognize_formula_pseudo_page(
            pseudo,
            client=client,
            batch_id=args.batch_id,
        )
        (out_dir / "paddle_response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_binding_outputs(out_dir, recognitions, manifest["parent_text"])

    print(f"wrote {pseudo_path}")
    print(f"wrote {out_dir / 'manifest.json'}")
    if not args.call_paddle and not args.response_json:
        print("offline mode: pass --call-paddle or --response-json to generate binding suggestions")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="file/未命名项目.ocrproj", help="project .ocrproj path")
    parser.add_argument("--project-id", type=int, default=None, help="project id; default latest")
    parser.add_argument("--page-index", type=int, default=0, help="0-based page index in selected project")
    parser.add_argument("--parent-index", type=int, default=3, help="layout row index to inspect")
    parser.add_argument("--out-dir", default="tmp/inline_formula_rebind", help="output directory")
    parser.add_argument("--response-json", default="", help="existing Paddle response json for offline binding")
    parser.add_argument("--call-paddle", action="store_true", help="submit pseudo page to configured Paddle API")
    parser.add_argument("--batch-id", default="", help="optional Paddle batch id")
    return parser.parse_args()


def _snapshot_project_db(project_path: Path, out_dir: Path) -> Path:
    snapshot = out_dir / "project_snapshot.ocrproj"
    shutil.copy2(project_path, snapshot)
    wal = Path(str(project_path) + "-wal")
    if wal.exists():
        shutil.copy2(wal, Path(str(snapshot) + "-wal"))
    shm = Path(str(project_path) + "-shm")
    if shm.exists():
        shutil.copy2(shm, Path(str(snapshot) + "-shm"))
    return snapshot


def _latest_project_id(store: ProjectStore) -> int:
    row = store.conn.execute("select id from project order by id desc limit 1").fetchone()
    if row is None:
        raise SystemExit("project database has no project rows")
    return int(row[0])


def _select_page(pages: list[Any], page_index: int):
    if page_index < 0 or page_index >= len(pages):
        raise SystemExit(f"page index out of range: {page_index}, pages={len(pages)}")
    return pages[page_index]


def _select_parent_row(rows: list[dict[str, Any]], parent_index: int) -> dict[str, Any]:
    if parent_index < 0 or parent_index >= len(rows):
        raise SystemExit(f"parent index out of range: {parent_index}, rows={len(rows)}")
    return rows[parent_index]


def _resolve_image_path(value: str, project_dir: Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_file():
        return path
    candidate = project_dir / path
    if candidate.is_file():
        return candidate
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    return path


def _write_recognition_outputs(
    out_dir: Path,
    pseudo,
    response: dict[str, Any],
    parent_text: str,
) -> None:
    (out_dir / "paddle_response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recognitions = recognitions_from_paddle_response(pseudo, response)
    _write_binding_outputs(out_dir, recognitions, parent_text)


def _write_binding_outputs(out_dir: Path, recognitions, parent_text: str) -> None:
    suggestions = suggest_formula_bindings(parent_text, recognitions)
    (out_dir / "recognitions.json").write_text(
        json.dumps([asdict(item) for item in recognitions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "binding_suggestions.json").write_text(
        json.dumps([asdict(item) for item in suggestions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'recognitions.json'}")
    print(f"wrote {out_dir / 'binding_suggestions.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
