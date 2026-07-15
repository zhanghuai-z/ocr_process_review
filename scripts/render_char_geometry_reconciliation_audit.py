#!/usr/bin/env python3
"""Render a read-only audit of production geometry reconciliation."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.geometry.char_reconciler import reconcile_char_geometry
from app.models.char_geometry import NativeGeometryProposal
from app.utils.image_io import read_cv_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(args.project)
    connection.row_factory = sqlite3.Row
    pages = connection.execute(
        "SELECT id, page_number, image_path FROM page ORDER BY page_number, id"
    ).fetchall()
    report: list[dict[str, object]] = []
    for page in pages:
        image_path = Path(page["image_path"])
        if not image_path.is_absolute():
            image_path = args.project.parent / image_path
        image = read_cv_image(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read project image: {image_path}")
        before = image.copy()
        after = image.copy()
        page_changes: list[dict[str, object]] = []
        lines = connection.execute(
            "SELECT l.id, l.uid, l.text, l.x, l.y, l.w, l.h "
            "FROM line l JOIN block b ON b.id=l.block_id "
            "WHERE b.page_id=? ORDER BY b.block_order, l.id",
            (page["id"],),
        ).fetchall()
        chars_by_line: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for char in connection.execute(
            "SELECT c.id, c.line_id, c.char, c.confidence, c.x, c.y, c.w, c.h "
            "FROM char_ c JOIN line l ON l.id=c.line_id "
            "JOIN block b ON b.id=l.block_id WHERE b.page_id=? ORDER BY c.id",
            (page["id"],),
        ).fetchall():
            chars_by_line[int(char["line_id"])].append(char)
        for line in lines:
            chars = chars_by_line.get(int(line["id"]), [])
            proposals = tuple(
                NativeGeometryProposal(
                    index,
                    (int(char["x"]), int(char["y"]), int(char["x"] + char["w"]), int(char["y"] + char["h"])),
                    float(char["confidence"] or 0.0),
                )
                for index, char in enumerate(chars)
                if None not in (char["x"], char["y"], char["w"], char["h"])
                and int(char["w"]) > 0
                and int(char["h"]) > 0
            )
            if len(proposals) < 2:
                continue
            line_bbox = (
                int(line["x"]),
                int(line["y"]),
                int(line["x"] + line["w"]),
                int(line["y"] + line["h"]),
            )
            page_height, page_width = image.shape[:2]
            left = max(0, min(page_width, line_bbox[0]))
            top = max(0, min(page_height, line_bbox[1]))
            right = max(left, min(page_width, line_bbox[2]))
            bottom = max(top, min(page_height, line_bbox[3]))
            crop = image[top:bottom, left:right]
            local_proposals = tuple(
                NativeGeometryProposal(
                    proposal.index,
                    (
                        proposal.bbox[0] - left,
                        proposal.bbox[1] - top,
                        proposal.bbox[2] - left,
                        proposal.bbox[3] - top,
                    ),
                    proposal.confidence,
                )
                for proposal in proposals
            )
            result = reconcile_char_geometry(crop, local_proposals)
            proposals_by_index = {proposal.index: proposal for proposal in proposals}
            merged = [atom for atom in result.atoms if len(atom.proposal_indices) > 1]
            if not merged:
                continue
            for atom in merged:
                member_text = "".join(str(chars[index]["char"] or "") for index in atom.proposal_indices)
                member_boxes = [proposals_by_index[index].bbox for index in atom.proposal_indices]
                for bbox in member_boxes:
                    cv2.rectangle(before, bbox[:2], bbox[2:], (0, 0, 255), 2)
                atom_bbox = (
                    atom.bbox[0] + left,
                    atom.bbox[1] + top,
                    atom.bbox[2] + left,
                    atom.bbox[3] + top,
                )
                cv2.rectangle(after, atom_bbox[:2], atom_bbox[2:], (0, 180, 0), 3)
                page_changes.append({
                    "line_uid": line["uid"],
                    "line_text": line["text"],
                    "member_text": member_text,
                    "proposal_indices": list(atom.proposal_indices),
                    "member_boxes": [list(bbox) for bbox in member_boxes],
                    "atom_bbox": list(atom_bbox),
                    "reason": atom.reason,
                })
        if not page_changes:
            continue
        page_dir = args.output_dir / f"page_{int(page['page_number']):03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(page_dir / "01_native_proposals.png"), before)
        cv2.imwrite(str(page_dir / "02_reconciled_atoms.png"), after)
        (page_dir / "changes.json").write_text(
            json.dumps(page_changes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report.append({
            "page_number": int(page["page_number"]),
            "change_count": len(page_changes),
            "changes": page_changes,
        })
    connection.close()
    summary = {
        "schema": "char_geometry_reconciliation_audit.v1",
        "project": str(args.project),
        "changed_pages": len(report),
        "changed_atoms": sum(int(page["change_count"]) for page in report),
        "pages": report,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ("changed_pages", "changed_atoms")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
