from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def _is_cjk(char: str) -> bool:
    if len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _compact(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def _display_char(row: sqlite3.Row) -> str:
    char = str(row["char"] or "")
    granularity = str(row["bbox_granularity"] or "")
    if granularity == "char" and len(char) == 1:
        return char
    return str(row["token_text"] or char)


def _load_rows(connection: sqlite3.Connection) -> tuple[list[sqlite3.Row], dict[int, list[sqlite3.Row]]]:
    lines = connection.execute(
        """
        SELECT l.*, b.block_type, b.block_order, p.page_number, p.uid AS page_uid
        FROM line l
        JOIN block b ON b.id = l.block_id
        JOIN page p ON p.id = b.page_id
        ORDER BY p.page_number, b.block_order, l.y, l.x, l.id
        """
    ).fetchall()
    chars_by_line: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM char_ ORDER BY line_id, x, y, id"):
        chars_by_line[int(row["line_id"])].append(row)
    return lines, chars_by_line


def audit(project: Path, minimum_center_gap_ratio: float) -> dict[str, object]:
    connection = sqlite3.connect(project)
    connection.row_factory = sqlite3.Row
    try:
        lines, chars_by_line = _load_rows(connection)
    finally:
        connection.close()

    accepted_lines = 0
    examined_pairs = 0
    candidates: list[dict[str, object]] = []
    rejected = defaultdict(int)

    for line in lines:
        chars = chars_by_line.get(int(line["id"]), [])
        if not chars:
            rejected["no_chars"] += 1
            continue
        if float(line["w"] or 0) < float(line["h"] or 0):
            rejected["non_horizontal"] += 1
            continue
        if any(str(char["bbox_granularity"] or "") != "char" for char in chars):
            rejected["non_char_carrier"] += 1
            continue
        glyphs = [_display_char(char) for char in chars]
        if any(len(glyph) != 1 for glyph in glyphs):
            rejected["non_single_glyph"] += 1
            continue
        source_text = str(line["ocr_text"] or line["text"] or "")
        joined = "".join(glyphs)
        if _compact(source_text) != _compact(joined):
            rejected["text_geometry_mismatch"] += 1
            continue

        heights = [float(char["h"] or 0) for char in chars if float(char["h"] or 0) > 0]
        if not heights:
            rejected["missing_height"] += 1
            continue
        median_height = statistics.median(heights)
        accepted_lines += 1
        inferred: list[str] = []
        inserted: list[dict[str, object]] = []
        for index, (glyph, char) in enumerate(zip(glyphs, chars)):
            inferred.append(glyph)
            if index + 1 >= len(chars):
                continue
            next_glyph = glyphs[index + 1]
            if not (_is_cjk(glyph) and _is_cjk(next_glyph)):
                continue
            examined_pairs += 1
            center = float(char["x"]) + float(char["w"]) / 2.0
            next_char = chars[index + 1]
            next_center = float(next_char["x"]) + float(next_char["w"]) / 2.0
            ratio = (next_center - center) / median_height
            if ratio < minimum_center_gap_ratio:
                continue
            inferred.append(" ")
            inserted.append(
                {
                    "pair": glyph + next_glyph,
                    "ratio": round(ratio, 3),
                    "gap_px": round(float(next_char["x"]) - (float(char["x"]) + float(char["w"])), 1),
                }
            )
        if inserted:
            candidates.append(
                {
                    "page": int(line["page_number"]),
                    "page_uid": line["page_uid"],
                    "block_type": line["block_type"],
                    "line_uid": line["uid"],
                    "original": joined,
                    "inferred": "".join(inferred),
                    "inserted": inserted,
                }
            )

    return {
        "schema": "intentional_cjk_spacing_audit.v1",
        "project": str(project),
        "minimum_center_gap_ratio": minimum_center_gap_ratio,
        "accepted_lines": accepted_lines,
        "examined_cjk_pairs": examined_pairs,
        "candidate_lines": len(candidates),
        "rejected_lines": dict(sorted(rejected.items())),
        "candidates": candidates,
    }


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# 68 页字间留白审计",
        "",
        f"- 候选整行：{report['candidate_lines']}",
        f"- 有效横排行：{report['accepted_lines']}",
        f"- 检查相邻汉字对：{report['examined_cjk_pairs']}",
        f"- 实验分界：字符中心距 / 行内中字高 >= {report['minimum_center_gap_ratio']}",
        "- 限制：只检查字符载体完整、OCR 文本与字符序列一致的横排行；不修改项目。",
        "",
    ]
    for item in report["candidates"]:
        insertions = "，".join(
            f"{entry['pair']}={entry['ratio']}x (空白 {entry['gap_px']}px)"
            for entry in item["inserted"]
        )
        lines.extend(
            [
                f"## 第 {item['page']} 页 · {item['block_type']}",
                "",
                f"- 原始：`{item['original']}`",
                f"- 推断：`{item['inferred']}`",
                f"- 插空：{insertions}",
                f"- line uid：`{item['line_uid']}`",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit intentional CJK spacing from persisted char geometry.")
    parser.add_argument("project", type=Path)
    parser.add_argument("--minimum-center-gap-ratio", type=float, default=2.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = audit(args.project, args.minimum_center_gap_ratio)
    (args.output_dir / "spacing_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(report, args.output_dir / "spacing_candidates.md")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
