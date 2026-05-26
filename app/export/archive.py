"""XML/JSON archive parity helpers."""
from __future__ import annotations

import json
from typing import Any

from lxml import etree


def json_archive_projection(data: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic fields that must align with the XML authority archive."""
    return {
        "version": data.get("version"),
        "project": data.get("project", {}),
        "pages": [
            {
                "page_number": page.get("page_number"),
                "source_image": page.get("source_image"),
                "source_meta": page.get("source_meta", {}),
                "size": page.get("size", {}),
                "status": page.get("status", ""),
                "elements": [_element_projection(element) for element in page.get("elements", [])],
            }
            for page in data.get("pages", [])
        ],
        "assets": [_asset_projection(asset) for asset in data.get("assets", [])],
        "diagnostics": [_diagnostic_projection(item) for item in data.get("diagnostics", [])],
    }


def xml_archive_projection(root_or_path) -> dict[str, Any]:
    """Parse XML authority archive into the same semantic projection as JSON."""
    root = etree.parse(str(root_or_path)).getroot() if not hasattr(root_or_path, "tag") else root_or_path
    project_el = root.find("Project")
    project = {
        "name": project_el.get("name", "") if project_el is not None else "",
        "page_count": _int(project_el.get("page_count")) if project_el is not None else 0,
        "summary": _fields(project_el.find("Summary") if project_el is not None else None),
    }
    if project_el is not None and project_el.get("created_at"):
        project["created_at"] = _float(project_el.get("created_at"))
    if project_el is not None and project_el.get("updated_at"):
        project["updated_at"] = _float(project_el.get("updated_at"))

    pages = []
    for page_el in root.findall("./Pages/Page"):
        pages.append({
            "page_number": _int(page_el.get("number")),
            "source_image": page_el.get("source_image", ""),
            "source_meta": _fields(page_el.find("SourceMeta")),
            "size": {
                "w": _int(page_el.get("width")),
                "h": _int(page_el.get("height")),
            },
            "status": page_el.get("status", ""),
            "elements": [_xml_element_projection(el) for el in page_el.findall("./Elements/Element")],
        })

    return {
        "version": root.get("version"),
        "project": project,
        "pages": pages,
        "assets": [_xml_asset_projection(el) for el in root.findall("./Assets/Asset")],
        "diagnostics": [_xml_diagnostic_projection(el) for el in root.findall("./Diagnostics/Diagnostic")],
    }


def _element_projection(element: dict[str, Any]) -> dict[str, Any]:
    source = dict(element.get("source", {}))
    source.setdefault("line_ids", [])
    source.setdefault("char_ids", [])
    proof = dict(element.get("proof", {}))
    proof.setdefault("flags", [])
    result = {
        "id": element.get("id"),
        "kind": element.get("kind"),
        "page": element.get("page"),
        "order": element.get("order"),
        "bbox": element.get("bbox"),
        "source": source,
        "proof": proof,
        "payload": element.get("payload", {}),
    }
    if element.get("fallback"):
        result["fallback"] = element["fallback"]
    return result


def _asset_projection(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": asset.get("id"),
        "kind": asset.get("kind"),
        "page_number": asset.get("page_number"),
        "bbox": asset.get("bbox"),
        "path": asset.get("path", ""),
        "mime": asset.get("mime", ""),
    }


def _diagnostic_projection(item: dict[str, Any]) -> dict[str, Any]:
    result = {
        "level": item.get("level"),
        "code": item.get("code"),
        "message": item.get("message"),
    }
    if item.get("element_id"):
        result["element_id"] = item.get("element_id")
    if item.get("details"):
        result["details"] = item.get("details")
    return result


def _xml_element_projection(element_el) -> dict[str, Any]:
    result = {
        "id": element_el.get("id"),
        "kind": element_el.get("kind"),
        "page": _int(element_el.get("page")),
        "order": _int(element_el.get("order")),
        "bbox": _bbox(element_el.find("BBox")),
        "source": _xml_source(element_el.find("Source")),
        "proof": _xml_proof(element_el.find("Proof")),
        "payload": _xml_payload(element_el.find("Payload")),
    }
    fallback = _xml_fallback(element_el.find("Fallback"))
    if fallback:
        result["fallback"] = fallback
    return result


def _xml_source(source_el) -> dict[str, Any]:
    if source_el is None:
        return {}
    return {
        "page_number": _int(source_el.get("page_number")),
        "block_ids": _id_list(source_el, "BlockIds", "BlockId"),
        "origin": source_el.get("origin", ""),
        "line_ids": _id_list(source_el, "LineIds", "LineId"),
        "char_ids": _id_list(source_el, "CharIds", "CharId"),
        "block_type": source_el.get("block_type", ""),
    }


def _xml_proof(proof_el) -> dict[str, Any]:
    if proof_el is None:
        return {}
    return {
        "status": proof_el.get("status", ""),
        "confidence": _float(proof_el.get("confidence")),
        "flags": [el.text or "" for el in proof_el.findall("./Flags/Flag")],
        "corrected": proof_el.get("corrected") == "true",
    }


def _xml_payload(payload_el) -> dict[str, Any]:
    if payload_el is None:
        return {}
    payload = _fields(payload_el)
    lines = []
    for line_el in payload_el.findall("./Lines/Line"):
        item = {
            "line_id": _typed_id(line_el.get("line_id")),
            "text": line_el.text or "",
            "bbox": _bbox(line_el.find("BBox")),
            "confidence": _float(line_el.get("confidence")),
            "status": line_el.get("status", ""),
        }
        ocr_el = line_el.find("OcrText")
        if ocr_el is not None:
            item["ocr_text"] = ocr_el.text or ""
        lines.append(item)
    if lines:
        payload["lines"] = lines
    return payload


def _xml_fallback(fallback_el) -> dict[str, Any]:
    if fallback_el is None:
        return {}
    result = {
        "used": fallback_el.get("used") == "true",
        "reason": fallback_el.get("reason", ""),
        "mode": fallback_el.get("mode", ""),
    }
    if fallback_el.get("asset_ref"):
        result["asset_ref"] = fallback_el.get("asset_ref")
    return result


def _xml_asset_projection(asset_el) -> dict[str, Any]:
    return {
        "id": asset_el.get("id"),
        "kind": asset_el.get("kind"),
        "page_number": _int(asset_el.get("page_number")),
        "bbox": _bbox(asset_el.find("BBox")),
        "path": asset_el.get("path", ""),
        "mime": asset_el.get("mime", ""),
    }


def _xml_diagnostic_projection(diagnostic_el) -> dict[str, Any]:
    result = {
        "level": diagnostic_el.get("level"),
        "code": diagnostic_el.get("code"),
        "message": diagnostic_el.text or "",
    }
    if diagnostic_el.get("element_id"):
        result["element_id"] = diagnostic_el.get("element_id")
    details = _fields(diagnostic_el.find("Details"))
    if details:
        result["details"] = details
    return result


def _fields(parent) -> dict[str, Any]:
    if parent is None:
        return {}
    result: dict[str, Any] = {}
    for field_el in parent.findall("Field"):
        name = field_el.get("name", "")
        value = field_el.text or ""
        type_name = field_el.get("type", "str")
        if type_name in {"dict", "list"}:
            result[name] = json.loads(value) if value else ([] if type_name == "list" else {})
        elif type_name == "int":
            result[name] = _int(value)
        elif type_name == "float":
            result[name] = _float(value)
        elif type_name == "bool":
            result[name] = value == "True"
        else:
            result[name] = value
    return result


def _bbox(bbox_el) -> dict[str, int] | None:
    if bbox_el is None:
        return None
    return {
        "x": _int(bbox_el.get("x")),
        "y": _int(bbox_el.get("y")),
        "w": _int(bbox_el.get("w")),
        "h": _int(bbox_el.get("h")),
    }


def _id_list(parent, container_tag: str, item_tag: str) -> list[int | str]:
    values: list[int | str] = []
    for item_el in parent.findall(f"./{container_tag}/{item_tag}"):
        values.append(_typed_id(item_el.text or ""))
    return values


def _typed_id(value: str | None) -> int | str:
    if value is None:
        return ""
    try:
        return int(value)
    except ValueError:
        return value


def _int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0
