import numpy as np

import app.engines.hanwang.micro_recblock as micro_module
from tests.charocr_native_route_fixture import run_micro_recblock_with_explicit_routes


def _gbk_code(char: str) -> int:
    return int.from_bytes(char.encode("gbk"), "little")


def test_large_horizontal_native_group_uses_one_normalized_physical_line(monkeypatch) -> None:
    seen_shapes: list[tuple[int, int]] = []

    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        assert recblocks_xyxy == [(10, 10, 190, 90)]
        return {
            "lines": [{
                "groups": [
                    {"bbox": {"left": 10, "top": 10, "right": 190, "bottom": 30}},
                    {"bbox": {"left": 10, "top": 10, "right": 190, "bottom": 90}},
                    {"bbox": {"left": 10, "top": 70, "right": 190, "bottom": 90}},
                ]
            }]
        }

    def fake_recog(image_bgr, **_kwargs):
        seen_shapes.append(tuple(image_bgr.shape[:2]))
        assert seen_shapes[-1] == (70, 137)
        return {
            "lines": [{
                "groups": [{
                    "bbox": {"left": 0, "top": 0, "right": 137, "bottom": 70},
                    "chars": [
                        {
                            "codes": [_gbk_code("标")],
                            "scores": [8],
                            "bbox": {"left": 10, "top": 14, "right": 50, "bottom": 56},
                        },
                        {
                            "codes": [_gbk_code("题")],
                            "scores": [8],
                            "bbox": {"left": 60, "top": 14, "right": 100, "bottom": 56},
                        },
                    ],
                }]
            }]
        }

    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_segimg", fake_segimg)
    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_recog", fake_recog)
    monkeypatch.setattr(micro_module, "_BATCH_DISABLED_FOR_SESSION", True)

    rows, stats = run_micro_recblock_with_explicit_routes(
        np.full((120, 220, 3), 255, dtype=np.uint8),
        [{
            "block_label": "title",
            "block_bbox": [10, 10, 190, 90],
            "block_content": "标题",
        }],
        include_chars=True,
    )

    assert seen_shapes == [(70, 137)]
    assert stats.n_groups == 1
    assert stats.recog_probe_calls == 1
    assert rows[0].text == "标题"
    assert rows[0].lines[0].bbox == (10, 10, 190, 90)
    assert rows[0].lines[0].bbox_source == "ppocrv6_physical_routing_line"
    assert rows[0].recog_group_bboxes == [(2, 0, 198, 100)]
    assert [char.bbox for char in rows[0].lines[0].chars] == [
        (16, 20, 74, 80),
        (88, 20, 145, 80),
    ]

    audits = rows[0].segimg_group_audits
    assert sum(item.get("native_recog_superseded_by") == "normalized_physical_line" for item in audits) == 3
    normalized = next(item for item in audits if item.get("native_input_mode") == "normalized_physical_line")
    assert normalized["native_input_core_height"] == 80
    assert normalized["native_input_core_height_target"] == 56
    assert normalized["native_input_original_shape"] == [100, 196]
    assert normalized["native_input_shape"] == [70, 137]


def test_large_horizontal_native_groups_keep_normalization_in_batch(monkeypatch) -> None:
    def fake_segimg(image_bgr, *, recblocks_xyxy=None, timeout=0):
        return {
            "lines": [
                {"groups": [{"bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}}]}
                for x1, y1, x2, y2 in recblocks_xyxy
            ]
        }

    seen_shapes: list[list[tuple[int, int]]] = []

    def fake_batch(crops, **_kwargs):
        seen_shapes.append([tuple(crop.shape[:2]) for crop in crops])
        return [
            {
                "lines": [{
                    "groups": [{
                        "bbox": {"left": 0, "top": 0, "right": 137, "bottom": 70},
                        "chars": [{
                            "codes": [_gbk_code(char)],
                            "scores": [8],
                            "bbox": {"left": 20, "top": 14, "right": 60, "bottom": 56},
                        }],
                    }]
                }]
            }
            for char in ("甲", "乙")
        ]

    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_segimg", fake_segimg)
    monkeypatch.setattr(micro_module.native_bridge, "run_linecut_recog_batch_list", fake_batch)
    monkeypatch.setattr(micro_module, "_BATCH_DISABLED_FOR_SESSION", False)

    rows, stats = run_micro_recblock_with_explicit_routes(
        np.full((230, 220, 3), 255, dtype=np.uint8),
        [
            {"block_label": "title", "block_bbox": [10, 10, 190, 90], "block_content": "甲"},
            {"block_label": "title", "block_bbox": [10, 130, 190, 210], "block_content": "乙"},
        ],
        include_chars=True,
    )

    assert seen_shapes == [[(70, 137), (70, 137)]]
    assert stats.recog_batch_chunks == 1
    assert stats.recog_batch_failures == 0
    assert [row.text for row in rows] == ["甲", "乙"]
    assert [row.lines[0].bbox for row in rows] == [
        (10, 10, 190, 90),
        (10, 130, 190, 210),
    ]
