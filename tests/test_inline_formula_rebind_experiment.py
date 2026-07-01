import numpy as np

from app.experimental.inline_formula_rebind import (
    build_formula_pseudo_page,
    formula_crop_bboxes_from_route_subblocks,
    formula_spans,
    recognitions_from_paddle_response,
    suggest_formula_bindings,
)


def test_formula_pseudo_page_stacks_crops_and_keeps_original_bboxes():
    image = np.full((100, 220, 3), 255, dtype=np.uint8)
    image[10:30, 40:90] = 0
    image[60:80, 120:180] = 0

    pseudo = build_formula_pseudo_page(
        image,
        [(40, 10, 90, 30), (120, 60, 180, 80)],
        crop_pad=4,
        row_gap=10,
        margin=8,
        min_row_height=32,
    )

    assert pseudo.image_bgr.shape[0] == 8 * 2 + 32 * 2 + 10
    assert len(pseudo.crops) == 2
    assert pseudo.crops[0].bbox == (40, 10, 90, 30)
    assert pseudo.crops[0].crop_bbox == (36, 6, 94, 34)
    assert pseudo.crops[0].pseudo_bbox == (8, 8, 66, 40)
    assert pseudo.crops[1].bbox == (120, 60, 180, 80)


def test_formula_crop_bboxes_split_generated_box_around_manual_formula():
    subblocks = [
        {"block_label": "inline_formula", "block_bbox": [445, 556, 884, 620]},
        {"block_label": "inline_formula", "block_bbox": [1242, 562, 1453, 620]},
        {
            "block_label": "inline_formula",
            "block_bbox": [680, 562, 883, 613],
            "_layout_manual_route_subblock": True,
        },
    ]

    bboxes = formula_crop_bboxes_from_route_subblocks(subblocks)

    assert bboxes == [
        (445, 556, 680, 620),
        (680, 562, 883, 613),
        (1242, 562, 1453, 620),
    ]


def test_formula_crop_recognition_binds_to_parent_formula_spans():
    image = np.full((900, 1500, 3), 255, dtype=np.uint8)
    parent_text = (
        "其中， $ GGF_{it}^{Post-short} $、 $ GGF_{it}^{Post-long} $ "
        "均为虚拟变量， $ GGF_{it}^{Post-short} $ 在企业获得政府引导基金"
    )
    pseudo = build_formula_pseudo_page(
        image,
        [(445, 556, 653, 620), (686, 565, 884, 614), (1242, 562, 1453, 620)],
    )

    records = []
    for crop, text in zip(
        pseudo.crops,
        [
            "$ GGF_{it}^{Post-short} $",
            "$ GGF_{it}^{Post-long} $",
            "$ GGF_{it}^{Post-short} $",
        ],
    ):
        x1, y1, x2, y2 = crop.pseudo_bbox
        records.append({
            "block_label": "inline_formula",
            "block_bbox": [x1 + 2, y1 + 2, x2 - 2, y2 - 2],
            "block_content": text,
        })
    response = {"result": {"layoutParsingResults": records}}

    spans = formula_spans(parent_text)
    recognitions = recognitions_from_paddle_response(pseudo, response)
    suggestions = suggest_formula_bindings(parent_text, recognitions)

    assert [span.text for span in spans] == [
        "$ GGF_{it}^{Post-short} $",
        "$ GGF_{it}^{Post-long} $",
        "$ GGF_{it}^{Post-short} $",
    ]
    assert [item.text for item in recognitions] == [
        "$ GGF_{it}^{Post-short} $",
        "$ GGF_{it}^{Post-long} $",
        "$ GGF_{it}^{Post-short} $",
    ]
    assert [item.span_index for item in suggestions] == [0, 1, 2]
    assert [item.status for item in suggestions] == [
        "duplicate_key_ordered",
        "exact",
        "duplicate_key_ordered",
    ]


def test_formula_crop_recognition_fuzzy_matches_missing_delimiters():
    image = np.full((60, 160, 3), 255, dtype=np.uint8)
    parent_text = "甲 $ GGF_{it}^{Post-long} $ 乙"
    pseudo = build_formula_pseudo_page(image, [(20, 10, 90, 40)])
    crop = pseudo.crops[0]
    x1, y1, x2, y2 = crop.pseudo_bbox
    response = {
        "result": {
            "layoutParsingResults": [
                {
                    "block_label": "text",
                    "block_bbox": [x1, y1, x2, y2],
                    "block_content": "GGF_{it}^{Post-long}",
                }
            ]
        }
    }

    recognitions = recognitions_from_paddle_response(pseudo, response)
    suggestions = suggest_formula_bindings(parent_text, recognitions)

    assert len(suggestions) == 1
    assert suggestions[0].span_index == 0
    assert suggestions[0].span_text == "$ GGF_{it}^{Post-long} $"
    assert suggestions[0].status == "exact"
