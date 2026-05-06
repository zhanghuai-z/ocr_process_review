"""OCR engine global config singleton.

Modes:
  local - local PaddleOCR model (lazy-loaded)
  api   - AiStudio cloud Serving API

Confirmed AiStudio API:
  POST {api_url}/layout-parsing
  Authorization: token <api_token>
  Body: {"file": "<base64_JPEG>", "fileType": 1}
  Response:
    result.layoutParsingResults[0].prunedResult:
      layout_det_res.boxes[]: {label, coordinate:[x1,y1,x2,y2]}
      overall_ocr_res: {rec_texts, rec_scores, rec_boxes:[[x1,y1,x2,y2],...]}
"""
from __future__ import annotations

_config: dict = {
    "mode": "local",
    "api_url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com",
    "api_timeout": 30,
    "api_token": "",
}


def get_config() -> dict:
    return _config


def update_config(**kwargs) -> None:
    for k, v in kwargs.items():
        if k in _config:
            _config[k] = v
