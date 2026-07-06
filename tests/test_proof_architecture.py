"""Proof architecture guardrails.

Proof panels are views/controllers for user intent. They must not become a
second persistence or model-mutation layer next to ProofEditService.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


PROOF_UI_DIR = Path("app/ui/proof")
APP_DIR = Path("app")

_FORBIDDEN_LINE_FACT_ATTRS = {
    "text",
    "final_text",
    "original_text",
    "ocr_text",
    "chars",
    "proof_status",
}

_FORBIDDEN_LINE_MUTATION_METHODS = {
    "update_text",
    "update_final_text",
    "ensure_text_contract",
}

_FORBIDDEN_UI_IMPORTS = {
    "app.core.project_store",
    "app.services.proof_persistence_service",
}

_FORBIDDEN_UI_IMPORT_NAMES = {
    "ProjectStore",
    "ProofPersistenceService",
    "save_displayed_edit_result",
}


def _proof_ui_sources() -> list[Path]:
    return sorted(PROOF_UI_DIR.glob("*.py"))


def _assigned_attr_targets(tree: ast.AST) -> list[ast.Attribute]:
    targets: list[ast.Attribute] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            raw_targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            raw_targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            raw_targets = [node.target]
        else:
            continue
        for target in raw_targets:
            targets.extend(_flatten_attr_targets(target))
    return targets


def _flatten_attr_targets(node: ast.AST) -> list[ast.Attribute]:
    if isinstance(node, ast.Attribute):
        return [node]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[ast.Attribute] = []
        for item in node.elts:
            out.extend(_flatten_attr_targets(item))
        return out
    return []


def _function_source(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    return ""


def test_proof_ui_does_not_directly_mutate_line_facts():
    offenders: list[str] = []
    for path in _proof_ui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr not in _FORBIDDEN_LINE_FACT_ATTRS:
                continue
            owner = ast.unparse(target.value)
            if owner in {"self", "cls"}:
                continue
            offenders.append(f"{path}:{target.lineno}: {owner}.{target.attr}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _FORBIDDEN_LINE_MUTATION_METHODS:
                continue
            owner = ast.unparse(func.value)
            offenders.append(f"{path}:{node.lineno}: {owner}.{func.attr}()")
    assert offenders == []


def test_proof_ui_uses_edit_service_instead_of_storage_or_probe_write_ports():
    offenders: list[str] = []
    for path in _proof_ui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _FORBIDDEN_UI_IMPORTS:
                    offenders.append(f"{path}:{node.lineno}: from {module} import ...")
                for alias in node.names:
                    if alias.name in _FORBIDDEN_UI_IMPORT_NAMES:
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_UI_IMPORTS:
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
    assert offenders == []


def test_retired_proof_cell_row_view_is_not_restored():
    assert not (PROOF_UI_DIR / "char_cell_row.py").exists()


def test_retired_core_proof_engine_is_not_restored():
    assert not Path("app/core/proof_engine.py").exists()


def test_retired_ocr_token_char_mapper_is_not_restored():
    assert not Path("app/core/token_char_mapper.py").exists()


def test_retired_legacy_raw_payload_splitter_is_not_restored():
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "split_legacy_raw_payload" in source or "LEGACY_RAW_APP_PAYLOAD_KEYS" in source:
            offenders.append(str(path))
    assert offenders == []


def test_retirement_plan_names_active_physical_projections():
    source = Path("compatibility-retirement-plan.md").read_text(encoding="utf-8")

    assert "当前没有登记中的兼容入口。" not in source
    assert "当前受控过渡投影" in source
    for projection in (
        "`Page.blocks`",
        "`Block.lines`",
        "`Line.chars`",
        "`Line.text` / `Line.ocr_text`",
        "`block.raw_payload_json` / `block.app_payload_json`",
    ):
        assert projection in source
    assert "新代码必须经过对应 boundary/helper 访问" in source


def test_current_truth_docs_do_not_reintroduce_stale_projection_terms():
    truth_source = Path("CURRENT_TRUTH_MAP.md").read_text(encoding="utf-8")
    retirement_source = Path("compatibility-retirement-plan.md").read_text(encoding="utf-8")
    baseline_source = Path("baseline-2026-06-12.md").read_text(encoding="utf-8")

    for stale in (
        "旧链路投影",
        "旧 UI/存储投影",
        "旧 `Page.blocks`",
        "旧 block tree",
        "旧 route dict API",
    ):
        assert stale not in truth_source
        assert stale not in retirement_source
    assert "归档说明" in baseline_source
    assert "CURRENT_TRUTH_MAP.md" in baseline_source


def test_fallback_strategy_register_covers_current_boundaries():
    source = Path("fallback-strategy-register.md").read_text(encoding="utf-8")

    for marker in (
        "_request_with_network_fallback",
        "MISSING_LINE_BBOX_FLAG",
        "ProofCropService.normalize_pages",
        "ensure_line_char_bboxes",
        "proof_fallback_warning",
        "is_char_index_hidden_geometry",
        "hanwang:CharRcg:char_fallback",
        "bind_latin_tokens_to_engcut_chars",
        "LATIN_ENGCUT_WORD_FALLBACK_STATUS",
        "latin_token_engcut_word_fallback",
        "equal_grid_fallback",
        "ExportFallback",
        "image_fallback",
    ):
        assert marker in source

    assert "PPVL 不再作为正文 OCR fallback" in source
    assert "不得因为正文切片 fallback 再被送入 Hanwang 普通文字识别" in source


def test_current_runtime_projection_terminology_is_not_described_as_legacy():
    files = [
        Path("app/models/ocr_observation.py"),
        Path("app/models/ocr_character_observation.py"),
        Path("app/models/ocr_character_observation_store.py"),
        Path("app/models/layout_snapshot_store.py"),
        Path("app/core/ocr_proof_projection.py"),
        Path("app/core/line_text_contract.py"),
        Path("app/core/paddle_v16_client.py"),
        Path("app/core/paddle_line_routing.py"),
        Path("app/core/layout_analyzer.py"),
        Path("app/core/app_config.py"),
        Path("app/core/api_profiles.py"),
    ]
    stale_terms = (
        "compatibility projection",
        "compatibility models",
        "legacy layout result envelope",
        "legacy route caches",
        "normalizing legacy",
        "兼容不同 Paddle",
        "兼容 API 返回",
        "历史配置字段",
        "旧 /layout-parsing",
    )
    offenders: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for term in stale_terms:
            if term in source:
                offenders.append(f"{path}: {term}")
    assert offenders == []


def test_retired_block_payload_helper_is_not_restored():
    assert not Path("app/core/block_payload.py").exists()


def test_ocr_dispatch_policy_uses_block_attributes_not_payload_guessing():
    source = Path("app/core/ocr_dispatch_policy.py").read_text(encoding="utf-8")
    assert "block_attributes(" in source
    assert "app_payload" not in source
    assert "raw_payload" not in source
    assert "block.source_label" not in source
    assert "PADDLE_BINDING_KEY" not in source


def test_page_scoped_ocr_pipeline_selection_uses_dispatch_plan():
    source = Path("app/services/ocr_pipeline.py").read_text(encoding="utf-8")
    required_functions = [
        "process_project",
        "_process_page_hybrid_work",
        "_process_page_with_hybrid_blocks",
        "_has_reusable_page_line_hints",
        "_reusable_page_line_hint_summary",
        "_assign_page_ocr_lines_to_blocks",
        "_assign_page_ocr_line_hints_to_blocks",
    ]
    for function_name in required_functions:
        function_source = _function_source(source, function_name)
        assert "build_text_ocr_dispatch_plan(page)" in function_source


def test_ocr_run_result_contract_is_not_defined_inside_pipeline():
    pipeline_source = Path("app/services/ocr_pipeline.py").read_text(encoding="utf-8")
    assert "class OcrProgress" not in pipeline_source
    assert "class OcrResult" not in pipeline_source
    assert "class PageOcrRunResult" not in pipeline_source
    contract_source = Path("app/services/ocr_run_result.py").read_text(encoding="utf-8")
    assert "class OcrProgress" in contract_source
    assert "class OcrRunResult" in contract_source
    assert "class PageOcrRunResult" in contract_source


def test_block_attributes_do_not_derive_semantics_from_payloads():
    source = Path("app/core/block_attributes.py").read_text(encoding="utf-8")
    assert "authoritative_paddle_label" not in source
    assert "paddle_binding" not in source
    assert "app_payload.get" not in source
    assert "raw_payload.get" not in source
    assert "raw_label and" not in source
    assert "app_payload:" not in source
    assert "raw_payload:" not in source
    assert 'payload["raw_payload"]' not in source


def test_export_ir_source_does_not_expose_raw_payload():
    ir_source = Path("app/export/ir.py").read_text(encoding="utf-8")
    tree = ast.parse(ir_source, filename="app/export/ir.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExportSource":
            class_source = ast.get_source_segment(ir_source, node) or ""
            assert "raw_payload" not in class_source
            break
    else:
        raise AssertionError("ExportSource class not found")

    builder_source = Path("app/export/ir_builder.py").read_text(encoding="utf-8")
    assert "raw_payload=dict(attrs.raw_payload)" not in builder_source

    xml_source = Path("app/export/xml.py").read_text(encoding="utf-8")
    assert "RawPayload" not in xml_source

    archive_source = Path("app/export/archive.py").read_text(encoding="utf-8")
    assert "RawPayload" not in archive_source


def test_app_payload_is_not_mutated_by_active_app_code():
    offenders: list[str] = []
    direct_mutation_patterns = (
        ".app_payload.pop(",
        ".app_payload.setdefault(",
        ".app_payload.update(",
    )
    assignment_pattern = re.compile(r"\.app_payload\[[^\]]+\]\s*=")
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for pattern in direct_mutation_patterns:
            if pattern in source:
                offenders.append(f"{path}: {pattern}")
        for match in assignment_pattern.finditer(source):
            offenders.append(f"{path}: {match.group(0)}")
    assert offenders == []


def test_app_payload_is_not_read_by_active_app_code():
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".app_payload.get(" in source:
            offenders.append(f"{path}: .app_payload.get(")
    assert offenders == []


def test_app_payload_access_stays_at_storage_validation_or_payload_boundary():
    allowed = {
        Path("app/core/model_validation.py"),
        Path("app/core/project_store.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        if ".app_payload" in source:
            offenders.append(str(path))
    assert offenders == []


def test_block_active_model_has_no_app_payload_field():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Block":
            block_source = ast.get_source_segment(source, node) or ""
            assert "app_payload:" not in block_source
            assert "raw_payload:" not in block_source
            break
    else:
        raise AssertionError("Block class not found")


def test_page_dispatch_policy_properties_are_not_restored():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Page":
            page_source = ast.get_source_segment(source, node) or ""
            assert "def text_blocks" not in page_source
            assert "def text_ocr_blocks" not in page_source
            break
    else:
        raise AssertionError("Page class not found")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OcrProject":
            project_source = ast.get_source_segment(source, node) or ""
            assert "def has_unrecognized_blocks" not in project_source
            break
    else:
        raise AssertionError("OcrProject class not found")


def test_page_and_project_ocr_observation_summary_properties_are_not_restored():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Page":
            page_source = ast.get_source_segment(source, node) or ""
            assert "def total_lines" not in page_source
            assert "def has_ocr_result" not in page_source
            break
    else:
        raise AssertionError("Page class not found")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OcrProject":
            project_source = ast.get_source_segment(source, node) or ""
            assert "def total_lines" not in project_source
            assert "def has_any_ocr_result" not in project_source
            assert "def all_pages_ocr_done" not in project_source
            assert "def has_any_ocr_done_page" not in project_source
            break
    else:
        raise AssertionError("OcrProject class not found")


def test_block_ocr_observation_summary_properties_are_not_restored():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Block":
            block_source = ast.get_source_segment(source, node) or ""
            assert "def avg_confidence" not in block_source
            break
    else:
        raise AssertionError("Block class not found")


def test_block_ocr_lines_access_goes_through_observation_boundary():
    allowed = {
        Path("app/models/ocr_observation.py"),
        Path("app/models/project.py"),
    }
    block_like_names = {
        "b",
        "blk",
        "block",
        "primary",
        "secondary",
        "source_block",
        "target_block",
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "lines":
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in block_like_names:
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
    source = Path("app/models/ocr_observation.py").read_text(encoding="utf-8")
    assert "ocr_lines_for_block(block, block.lines)" in source
    assert "set_ocr_lines_for_block(block, projected)" in source


def test_page_layout_blocks_access_goes_through_projection_boundary():
    allowed = {
        Path("app/models/layout_projection.py"),
        Path("app/models/project.py"),
    }
    page_like_names = {
        "page",
        "pg",
        "source_page",
        "target_page",
        "proof_page",
        "current_page",
        "out_page",
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed or path.parts[:2] == ("app", "ui"):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "blocks":
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in page_like_names:
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_layout_block_view_boundary_is_used_by_migrated_consumers():
    view_consumers = {
        Path("app/models/ocr_observation.py"),
        Path("app/services/ocr_dispatch_plan.py"),
        Path("app/engines/hanwang/micro_recblock.py"),
    }
    for path in view_consumers:
        source = path.read_text(encoding="utf-8")
        assert "layout_block_view" in source

    projection_bridges = {
        Path("app/core/project_store.py"),
        Path("app/services/layout_edit_service.py"),
        Path("app/services/ocr_pipeline.py"),
    }
    for path in projection_bridges:
        source = path.read_text(encoding="utf-8")
        assert "app.models.layout_projection" in source or "from .layout_projection" in source


def test_layout_snapshot_docs_do_not_name_current_projection_legacy():
    checked = {
        Path("app/services/layout_snapshot.py"),
        Path("app/models/layout_snapshot.py"),
        Path("app/models/layout_snapshot_projection.py"),
        Path("app/models/layout_snapshot_store.py"),
        Path("app/models/ocr_observation_store.py"),
    }
    offenders = [
        str(path)
        for path in checked
        if "legacy" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


def test_line_ocr_chars_access_goes_through_character_observation_boundary():
    allowed = {
        Path("app/models/ocr_character_observation.py"),
        Path("app/models/project.py"),
    }
    line_like_names = {
        "line",
        "candidate",
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if (
            path in allowed
            or path.parts[:2] == ("app", "ui")
            or path.parts[:2] == ("app", "engines")
        ):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "chars":
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in line_like_names:
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
    source = Path("app/models/ocr_character_observation.py").read_text(encoding="utf-8")
    assert "ocr_chars_for_line(line, line.chars)" in source
    assert "set_ocr_chars_for_line(line, projected)" in source


def test_character_observation_boundary_is_used_by_core_consumers():
    required_sources = {
        Path("app/core/char_bbox_utils.py"),
        Path("app/core/project_store.py"),
        Path("app/core/proof_atom.py"),
        Path("app/services/char_index_service.py"),
        Path("app/services/proof_crop_service.py"),
        Path("app/services/proof_probe_text_service.py"),
        Path("app/export/ir_builder.py"),
    }
    for path in required_sources:
        source = path.read_text(encoding="utf-8")
        assert "app.models.ocr_character_observation" in source


def test_carrier_detection_uses_proof_char_text_contract():
    service_paths = {
        Path("app/services/char_index_service.py"),
        Path("app/services/proof_crop_service.py"),
        Path("app/services/proof_probe_text_service.py"),
    }
    for path in service_paths:
        source = path.read_text(encoding="utf-8")
        assert "is_display_carrier" in source
        assert "def _is_tokenized_char" not in source
        assert 'bbox_granularity == "word" or len(char.char or "") > 1' not in source


def test_proof_geometry_quality_is_shared_by_proof_services():
    helper_source = Path("app/core/proof_geometry_quality.py").read_text(encoding="utf-8")
    assert "def is_estimated_or_unavailable_geometry" in helper_source
    assert "def is_char_index_hidden_geometry" in helper_source
    assert "source.startswith(\"hanwang:\")" in helper_source

    char_index_source = Path("app/services/char_index_service.py").read_text(encoding="utf-8")
    crop_source = Path("app/services/proof_crop_service.py").read_text(encoding="utf-8")
    assert "is_char_index_hidden_geometry" in char_index_source
    assert "is_estimated_or_unavailable_geometry" in crop_source
    assert "source.startswith(\"hanwang:\")" not in char_index_source
    assert "source in {\"fallback\", \"unavailable\"}" not in crop_source
    assert "granularity in {\"fallback\", \"unavailable\", \"line\"}" not in crop_source


def test_ocr_observation_geometry_writes_stay_at_observation_boundaries():
    allowed_by_pattern = {
        r"\bline\.bbox\s*=(?!=)": {Path("app/models/ocr_observation.py")},
        r"\bchar\.bbox\s*=(?!=)": {Path("app/models/ocr_character_observation.py")},
        r"\bline\.chars\s*=(?!=)": {Path("app/models/ocr_character_observation.py")},
        r"\bline\.chars\[[^\n]+\]\s*=(?!=)": {Path("app/models/ocr_character_observation.py")},
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for pattern, allowed_paths in allowed_by_pattern.items():
            if path in allowed_paths:
                continue
            for match in re.finditer(pattern, source):
                line_no = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{path}:{line_no}: {match.group(0)}")
    assert offenders == []

    assert "def set_ocr_line_bbox" in Path("app/models/ocr_observation.py").read_text(
        encoding="utf-8"
    )
    char_boundary_source = Path("app/models/ocr_character_observation.py").read_text(
        encoding="utf-8"
    )
    assert "def set_ocr_char_bbox" in char_boundary_source
    assert "def replace_line_ocr_char_span" in char_boundary_source


def test_ocr_line_bbox_reads_go_through_observation_boundary():
    allowed = {
        Path("app/models/ocr_observation.py"),
        # These use the typed RoutingLine/PaddleRouteLineHint DTOs, not app.models.Line.
        Path("app/core/layout_routing_contract.py"),
        Path("app/core/paddle_line_routing.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if (
            path in allowed
            or path.parts[:2] == ("app", "ui")
            or path.parts[:2] == ("app", "engines")
        ):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "bbox":
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id == "line":
                offenders.append(f"{path}:{node.lineno}: line.bbox")
    assert offenders == []


def test_workflow_page_state_reads_go_through_page_state_boundary():
    allowed = {
        Path("app/models/page_state.py"),
        Path("app/models/project.py"),
    }
    forbidden_attrs = {"is_analyzed", "is_ocr_done", "needs_ocr_rerun"}
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed or path.parts[:2] == ("app", "ui"):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in forbidden_attrs:
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in {"page", "p"}:
                offenders.append(f"{path}:{node.lineno}: {owner.id}.{node.attr}")
    assert offenders == []


def test_page_model_does_not_restore_workflow_interpretation_properties():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Page":
            page_source = ast.get_source_segment(source, node) or ""
            assert "def is_analyzed" not in page_source
            assert "def is_ocr_done" not in page_source
            assert "def needs_ocr_rerun" not in page_source
            break
    else:
        raise AssertionError("Page class not found")


def test_page_workflow_status_and_error_reads_go_through_page_state_boundary():
    allowed = {
        Path("app/models/page_state.py"),
        Path("app/models/project.py"),
        Path("app/models/ocr_observation.py"),
        Path("app/core/project_store.py"),
        # This reads ExportPage.status, not app.models.Page.status.
        Path("app/export/xml.py"),
    }
    checked_roots = [
        Path("app/core"),
        Path("app/services"),
        Path("app/controllers"),
        Path("app/export"),
        Path("app/ui"),
    ]
    forbidden_attrs = {"error_message", "ocr_invalidated_reason", "status"}
    page_like_names = {"page", "p"}
    offenders: list[str] = []
    for root in checked_roots:
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and node.attr in forbidden_attrs
                ):
                    continue
                owner = node.value
                if isinstance(owner, ast.Name) and owner.id in page_like_names:
                    offenders.append(f"{path}:{node.lineno}: {owner.id}.{node.attr}")
    assert offenders == []


def test_production_code_does_not_construct_line_with_chars_storage():
    allowed = {
        Path("app/models/project.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed or path.parts[:2] == ("app", "ui"):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "Line":
                continue
            if any(keyword.arg == "chars" for keyword in node.keywords):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_ocr_text_observation_boundary_is_used_by_producers():
    required_sources = {
        Path("app/core/ocr_proof_projection.py"),
        Path("app/engines/hanwang/micro_recblock.py"),
        Path("app/engines/fake_ocr_engine.py"),
        Path("app/engines/real_ocr_adapter.py"),
        Path("app/core/paddle_artifact_index.py"),
    }
    for path in required_sources:
        source = path.read_text(encoding="utf-8")
        assert "app.models.ocr_text_observation" in source
    boundary_source = Path("app/models/ocr_text_observation.py").read_text(encoding="utf-8")
    store_source = Path("app/models/ocr_text_observation_store.py").read_text(encoding="utf-8")
    line_contract_source = Path("app/core/line_text_contract.py").read_text(encoding="utf-8")
    assert "ocr_text_observation_for_line(" in boundary_source
    assert "set_ocr_text_observation_for_line(" in boundary_source
    assert "_TEXT_BY_LINE_OBJECT" in store_source
    assert "line_ocr_text_observation(line)" in line_contract_source


def test_ocr_review_flags_go_through_text_observation_boundary():
    required_sources = {
        Path("app/core/char_bbox_utils.py"),
        Path("app/core/ocr_line_hints.py"),
        Path("app/core/proof_line_facts.py"),
        Path("app/core/proof_line_utils.py"),
        Path("app/core/project_store.py"),
        Path("app/export/ir_builder.py"),
        Path("app/services/char_index_service.py"),
        Path("app/services/proof_crop_service.py"),
    }
    for path in required_sources:
        source = path.read_text(encoding="utf-8")
        assert "app.models.ocr_text_observation" in source

    allowed = {
        Path("app/models/project.py"),
        Path("app/models/ocr_text_observation.py"),
    }
    checked_roots = [
        Path("app/core"),
        Path("app/services"),
        Path("app/export"),
        Path("app/controllers"),
    ]
    offenders: list[str] = []
    for root in checked_roots:
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            source = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\bline\.review_flags\b", source):
                line_no = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{path}:{line_no}")
    assert offenders == []


def test_production_code_uses_ocr_text_observation_creator_for_line_text_fields():
    allowed = {
        Path("app/models/project.py"),
        Path("app/models/ocr_text_observation.py"),
        Path("app/core/project_store.py"),
    }
    text_fields = {"text", "ocr_text", "final_text", "original_text"}
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed or path.parts[:2] == ("app", "ui"):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "Line":
                continue
            used = sorted(keyword.arg for keyword in node.keywords if keyword.arg in text_fields)
            if used:
                offenders.append(f"{path}:{node.lineno}: {','.join(used)}")
    assert offenders == []


def test_production_code_does_not_construct_block_with_lines_storage():
    allowed = {
        Path("app/models/project.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "Block":
                continue
            if any(keyword.arg == "lines" for keyword in node.keywords):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_production_code_does_not_construct_page_with_blocks_storage():
    allowed = {
        Path("app/models/project.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "Page":
                continue
            if any(keyword.arg == "blocks" for keyword in node.keywords):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_recognize_ui_uses_layout_and_char_observation_boundaries():
    recognize_sources = [
        Path("app/ui/recognize/ocr_panel.py"),
        Path("app/ui/recognize/layout_panel.py"),
    ]
    offenders: list[str] = []
    for path in recognize_sources:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"blocks", "chars"}:
                offenders.append(f"{path}:{node.lineno}: .{node.attr}")
    assert offenders == []

    ocr_source = Path("app/ui/recognize/ocr_panel.py").read_text(encoding="utf-8")
    assert "app.models.layout_block_view" in ocr_source
    assert "app.models.layout_projection" not in ocr_source
    assert "block_ocr_line_observations" in ocr_source
    assert "block_ocr_lines" not in ocr_source
    assert "block_ocr_line_count" not in ocr_source
    assert "block_avg_confidence" not in ocr_source
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    assert "app.models.layout_projection" in layout_source
    assert "app.models.ocr_character_observation" in layout_source
    assert "block_ocr_line_observations" in layout_source
    assert "iter_page_ocr_line_observation_occurrences" in layout_source
    assert "block_ocr_lines" not in layout_source
    assert "page_ocr_line_count" not in layout_source
    assert "block_avg_confidence" not in layout_source


def test_image_viewer_reads_block_confidence_from_ocr_observation_store():
    source = Path("app/ui/widgets/image_viewer.py").read_text(encoding="utf-8")

    assert "block_ocr_line_observations" in source
    assert "line_ocr_confidence" in source
    assert "block_avg_confidence" not in source


def test_proof_ui_does_not_reach_legacy_layout_or_char_storage_directly():
    offenders: list[str] = []
    for path in sorted(PROOF_UI_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"blocks", "chars"}:
                offenders.append(f"{path}:{node.lineno}: .{node.attr}")
    assert offenders == []


def test_proof_ui_reads_ocr_lines_from_observation_store():
    targets = [
        Path("app/ui/proof/h_proof.py"),
        Path("app/ui/proof/v_proof.py"),
    ]

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "block_ocr_lines" not in source
        assert "block_has_ocr_lines" not in source
        assert "block_ocr_line_count" not in source
        assert "line_belongs_to_block" not in source

    assert "block_ocr_line_observations" in Path("app/ui/proof/h_proof.py").read_text(encoding="utf-8")
    assert "block_ocr_line_observations" in Path("app/ui/proof/v_proof.py").read_text(encoding="utf-8")


def test_line_text_facts_access_goes_through_text_contract_boundary():
    allowed = {
        Path("app/core/line_text_contract.py"),
        Path("app/models/ocr_text_observation.py"),
        Path("app/models/project.py"),
    }
    forbidden_attrs = {"text", "ocr_text", "confidence", "final_text", "original_text"}
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if (
            path in allowed
            or path.parts[:2] == ("app", "ui")
            or path.parts[:2] == ("app", "engines")
        ):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in forbidden_attrs:
                continue
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id == "line":
                offenders.append(f"{path}:{node.lineno}: line.{node.attr}")
    assert offenders == []


def test_proof_line_occurrence_lookup_stays_in_ocr_observation_boundary():
    probe_source = Path("app/services/proof_probe_text_service.py").read_text(encoding="utf-8")
    persist_source = Path("app/services/proof_persistence_service.py").read_text(encoding="utf-8")

    assert "def resolve_block_line_index" not in probe_source
    assert "find_block_ocr_line_index" in probe_source
    assert "def _iter_project_lines" not in persist_source
    assert "iter_project_ocr_line_occurrences" in persist_source


def test_proof_refresh_signatures_use_line_signature_contract():
    controller_source = Path("app/controllers/workflow_controller.py").read_text(encoding="utf-8")
    controller_signature_source = _function_source(controller_source, "_proof_pages_signature")
    assert "line_signature(line)" in controller_signature_source
    for forbidden in (
        "iter_line_ocr_char_occurrences",
        "char.token_text",
        "char.confidence",
        "char.bbox_source",
        "char.bbox_granularity",
    ):
        assert forbidden not in controller_signature_source

    vproof_source = Path("app/ui/proof/v_proof.py").read_text(encoding="utf-8")
    char_index_signature_source = _function_source(vproof_source, "_char_index_page_signature")
    assert "line_signature(line)" in char_index_signature_source
    for forbidden in (
        "line_ocr_chars(line)",
        "char.token_text",
        "char.confidence",
        "char.bbox_source",
        "char.bbox_granularity",
    ):
        assert forbidden not in char_index_signature_source


def test_vproof_uses_char_entry_display_contract():
    vproof_source = Path("app/ui/proof/v_proof.py").read_text(encoding="utf-8")
    service_source = Path("app/services/char_index_service.py").read_text(encoding="utf-8")

    assert "def char_entry_display_text" in service_source
    assert "char_entry_display_text" in vproof_source
    assert "entry.token_text or entry.char" not in vproof_source
    assert "(entry.token_text or entry.char)" not in vproof_source


def test_hproof_formula_debug_uses_char_display_contract():
    hproof_source = Path("app/ui/proof/h_proof.py").read_text(encoding="utf-8")
    assert "char_display_text" in hproof_source
    assert 'getattr(char, "token_text"' not in hproof_source


def test_proof_fallback_warning_is_owned_by_proof_crop_service():
    controller_source = Path("app/controllers/workflow_controller.py").read_text(encoding="utf-8")
    service_source = Path("app/services/proof_crop_service.py").read_text(encoding="utf-8")

    assert "def _proof_fallback_warning" not in controller_source
    assert "line_ocr_chars" not in controller_source
    assert "proof_fallback_warning(" in controller_source
    assert "def proof_fallback_warning" in service_source


def test_ui_char_display_uses_text_contract_helper():
    ui_sources = {
        Path("app/ui/widgets/image_viewer.py"),
        Path("app/ui/recognize/layout_panel.py"),
    }
    for path in ui_sources:
        source = path.read_text(encoding="utf-8")
        assert "char_display_text" in source
        assert "char.char or char.token_text" not in source


def test_raw_payload_is_not_mutated_directly_by_app_code():
    offenders: list[str] = []
    direct_mutation_patterns = (
        ".raw_payload.pop(",
        ".raw_payload.setdefault(",
        ".raw_payload.update(",
    )
    assignment_pattern = re.compile(r"\.raw_payload\[[^\]]+\]\s*=")
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for pattern in direct_mutation_patterns:
            if pattern in source:
                offenders.append(f"{path}: {pattern}")
        for match in assignment_pattern.finditer(source):
            offenders.append(f"{path}: {match.group(0)}")
    assert offenders == []


def test_raw_payload_constructor_writes_stay_at_storage_boundary():
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "raw_payload=" in source:
            offenders.append(str(path))
    assert offenders == []


def test_raw_payload_reads_stay_at_storage_validation_or_raw_artifact_boundary():
    allowed = {
        Path("app/core/model_validation.py"),
        Path("app/core/project_store.py"),
        Path("app/core/raw_ocr_artifact.py"),
    }
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        if "block.raw_payload" in source or '.raw_payload.get(' in source:
            offenders.append(str(path))
    assert offenders == []


def test_current_store_schema_does_not_persist_retired_block_payload_columns():
    source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    validation_source = Path("app/core/model_validation.py").read_text(encoding="utf-8")
    ddl_source = source.split("MIGRATIONS:", 1)[0]
    save_block_source = _function_source(source, "_save_block")
    load_blocks_source = _function_source(source, "_load_blocks")

    for retired in ("raw_payload_json", "app_payload_json"):
        assert f"{retired} TEXT" not in ddl_source
        assert retired not in save_block_source
        assert retired not in load_blocks_source
    assert "def _migrate_v23_drop_retired_block_payload_columns" in source
    assert "validate_persistent_block_payloads" not in validation_source


def test_export_semantic_filters_do_not_read_raw_payload_labels():
    markdown_source = Path("app/export/markdown.py").read_text(encoding="utf-8")
    markdown_tree = ast.parse(markdown_source, filename="app/export/markdown.py")
    for node in ast.walk(markdown_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_element_labels":
            fn_source = ast.get_source_segment(markdown_source, node) or ""
            assert "raw_payload" not in fn_source
            assert "block_label" not in fn_source
            break
    else:
        raise AssertionError("_element_labels function not found")

    pdf_source = Path("app/export/pdf.py").read_text(encoding="utf-8")
    pdf_tree = ast.parse(pdf_source, filename="app/export/pdf.py")
    for node in ast.walk(pdf_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_is_inline_formula_element":
            fn_source = ast.get_source_segment(pdf_source, node) or ""
            assert "raw_payload" not in fn_source
            assert "block_label" not in fn_source
            break
    else:
        raise AssertionError("_is_inline_formula_element function not found")


def test_table_text_layer_html_source_does_not_scan_app_payload():
    source = Path("app/services/table_text_layer_service.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/services/table_text_layer_service.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_table_html":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "app_payload" not in fn_source
            assert 'get("text")' not in fn_source
            assert "raw_block_text_values(" in fn_source
            break
    else:
        raise AssertionError("_table_html function not found")


def test_hanwang_formula_crop_targets_do_not_read_payload_labels():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/engines/hanwang/micro_recblock.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_inline_formula_crop_ocr_targets":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "app_payload" not in fn_source
            assert "raw_payload" not in fn_source
            assert "block_label" not in fn_source
            break
    else:
        raise AssertionError("_inline_formula_crop_ocr_targets function not found")


def test_hanwang_block_row_does_not_recover_parent_from_block_payloads():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/engines/hanwang/micro_recblock.py")
    helper_source = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_route_source_label_from_view":
            helper_source = ast.get_source_segment(source, node) or ""
            break
    assert "route_source_label(block)" in helper_source
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_layout_row_from_block":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "_route_source_label_from_view(block, view)" in fn_source
            assert "authoritative_paddle_label(raw_payload)" not in fn_source
            assert 'app_payload.get("_layout_paddle_parent_index"' not in fn_source
            assert 'raw_payload.get("_layout_paddle_parent_index"' not in fn_source
            assert 'app_payload.get("block_label"' not in fn_source
            break
    else:
        raise AssertionError("_layout_row_from_block function not found")


def test_hanwang_runtime_routing_does_not_read_block_source_label_directly():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/engines/hanwang/micro_recblock.py")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "source_label"
            and isinstance(node.ctx, ast.Load)
            and ast.unparse(node.value) == "block"
        ):
            offenders.append(f"line {node.lineno}: block.source_label")
    assert offenders == []


def test_hanwang_raw_payload_parent_matching_ignores_runtime_parent_index():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/engines/hanwang/micro_recblock.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_parent_index_for_raw_payload":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "_layout_paddle_parent_index" not in fn_source
            assert "paddle_parent_index" not in fn_source
            break
    else:
        raise AssertionError("_parent_index_for_raw_payload function not found")


def test_project_model_does_not_own_proof_export_summary():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    for retired_name in (
        "def full_text",
        "def proofed_lines",
        "def flagged_lines",
        "def total_flagged_lines",
        "def total_unproofed_lines",
        "def get_export_summary",
    ):
        assert retired_name not in source


def test_block_type_enum_does_not_own_paddle_label_mapping():
    source = Path("app/models/enums.py").read_text(encoding="utf-8")
    assert "from_paddle" not in source
    assert "map_paddle_label" not in source
    assert "paragraph_text" not in source
    assert "doc_title" not in source
    assert "inline_formula" not in source
    assert "formula_number" not in source
    assert "table_cell" not in source

    adapter_source = Path("app/adapters/paddle/layout_importer.py").read_text(encoding="utf-8")
    assert "map_paddle_label_to_block_type" in adapter_source
    assert "_PADDLE_LABEL_TO_BLOCK_TYPE" in adapter_source
    for forbidden in (
        '"title" in normalized',
        '"caption" in normalized',
        '"table" in normalized',
        '"formula" in normalized',
        '"reference" in normalized',
        '"paragraph" in normalized',
        "any(token in normalized",
        "normalized.startswith",
    ):
        assert forbidden not in adapter_source


def test_block_source_semantics_are_centralized():
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def is_user_authored_layout_block" in helper_source
    assert "def export_origin_for_block" in helper_source
    assert "def mark_layout_block_manual_draw" in helper_source
    assert "def mark_layout_block_user_edited" in helper_source

    for path in (
        Path("app/core/block_attributes.py"),
        Path("app/export/ir_builder.py"),
        Path("app/engines/hanwang/micro_recblock.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "BlockSource.MANUAL_DRAW" not in source
        assert "BlockSource.USER_EDITED" not in source
        assert "BlockSource.AUTO_TIGHTENED" not in source
        assert "block.source not in" not in source
        assert "block.source in" not in source
        assert "getattr(block.source" not in source
        assert "str(block.source" not in source

    assert "is_user_authored_layout_block(block)" in Path("app/core/block_attributes.py").read_text(encoding="utf-8")
    assert "export_origin_for_block(block)" in Path("app/export/ir_builder.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    assert "is_user_authored_layout_block(block)" in hanwang_source
    assert "block_source_value(block)" in hanwang_source

    layout_edit_source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    assert "mark_layout_block_manual_draw(new_block)" in layout_edit_source
    assert "mark_layout_block_user_edited(block)" in layout_edit_source
    assert "mark_layout_block_user_edited(primary)" in layout_edit_source
    assert "BlockSource.MANUAL_DRAW" not in layout_edit_source
    assert "BlockSource.USER_EDITED" not in layout_edit_source


def test_user_layout_source_writes_go_through_layout_block_state_helper():
    allowed = {Path("app/models/layout_block_state.py")}
    forbidden_sources = {
        "BlockSource.MANUAL_DRAW",
        "BlockSource.USER_EDITED",
        "BlockSource.AUTO_TIGHTENED",
    }
    offenders: list[str] = []

    def contains_forbidden_source(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and ast.unparse(child) in forbidden_sources:
                return True
        return False

    for path in APP_DIR.rglob("*.py"):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                targets = []
                value = None
            if value is not None and contains_forbidden_source(value):
                for target in targets:
                    for attr in _flatten_attr_targets(target):
                        if attr.attr == "source":
                            offenders.append(f"{path}:{node.lineno}: direct layout source assignment")
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "source" and contains_forbidden_source(keyword.value):
                        offenders.append(f"{path}:{node.lineno}: direct layout source constructor keyword")

    assert offenders == []


def test_layout_ocr_policy_writes_go_through_layout_block_state_helper():
    allowed = {Path("app/models/layout_block_state.py")}
    offenders: list[str] = []
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def set_layout_block_ocr_policy" in helper_source

    for path in APP_DIR.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr == "ocr_policy":
                offenders.append(f"{path}:{target.lineno}: direct ocr_policy assignment")

    assert offenders == []


def test_layout_source_label_writes_go_through_layout_block_state_helper():
    allowed = {Path("app/models/layout_block_state.py")}
    offenders: list[str] = []
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def set_layout_block_source_label" in helper_source

    for path in APP_DIR.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr == "source_label":
                offenders.append(f"{path}:{target.lineno}: direct source_label assignment")

    assert offenders == []


def test_layout_block_type_writes_go_through_layout_block_state_helper():
    allowed = {Path("app/models/layout_block_state.py")}
    offenders: list[str] = []
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def set_layout_block_type" in helper_source

    for path in APP_DIR.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr == "block_type":
                offenders.append(f"{path}:{target.lineno}: direct block_type assignment")

    assert offenders == []


def test_layout_block_note_writes_go_through_layout_block_state_helper():
    allowed = {Path("app/models/layout_block_state.py")}
    offenders: list[str] = []
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def set_layout_block_note" in helper_source
    assert "def append_layout_block_note_once" in helper_source

    for path in APP_DIR.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr == "note":
                offenders.append(f"{path}:{target.lineno}: direct note assignment")

    assert offenders == []


def test_layout_block_geometry_and_order_writes_go_through_layout_block_state_helper():
    helper_source = Path("app/models/layout_block_state.py").read_text(encoding="utf-8")
    assert "def set_layout_block_bbox" in helper_source
    assert "def set_layout_block_order" in helper_source

    direct_patterns = (
        r"\bblock\.bbox\s*=(?!=)",
        r"\bprimary\.bbox\s*=(?!=)",
        r"\bnew_block\.bbox\s*=(?!=)",
        r"\bbi\._block\.bbox\s*=(?!=)",
        r"\bself\._block\.bbox\s*=(?!=)",
        r"\bblock\.order\s*=(?!=)",
        r"\bprimary\.order\s*=(?!=)",
        r"\bnew_block\.order\s*=(?!=)",
    )
    offenders: list[str] = []
    for path in APP_DIR.rglob("*.py"):
        if path == Path("app/models/layout_block_state.py"):
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in direct_patterns:
            if re.search(pattern, source):
                offenders.append(f"{path}: {pattern}")

    assert offenders == []


def test_project_store_exposes_only_scoped_proof_line_write_port():
    from app.core.project_store import ProjectStore

    assert hasattr(ProjectStore, "update_proof_lines")
    assert not hasattr(ProjectStore, "update_line")
    assert not hasattr(ProjectStore, "update_line_with_chars")
    assert not hasattr(ProjectStore, "update_lines")


def test_project_store_does_not_mutate_line_text_contract_on_save():
    source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    assert "ensure_line_text_contract" not in source
    assert "line_text_contract(" in source


def test_project_store_save_block_has_no_layout_runtime_side_effects():
    source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    save_block_source = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_save_block":
            save_block_source = ast.get_source_segment(source, node) or ""
            break

    assert save_block_source
    assert "_strip_runtime" not in source
    assert "strip_runtime" not in save_block_source
    assert ".lines = []" not in save_block_source
    assert "mark_ocr_text_invalidated(" not in save_block_source
    assert "ocr_invalidated_reason =" not in save_block_source
    assert "raw_payload_json" not in save_block_source
    assert "app_payload_json" not in save_block_source
    assert "validate_block_model(block)" in save_block_source


def test_proof_line_facts_reads_ocr_text_through_contract():
    source = Path("app/core/proof_line_facts.py").read_text(encoding="utf-8")
    assert "line_text_contract(line)" in source
    assert 'getattr(line, "text"' not in source
    assert 'getattr(line, "ocr_text"' not in source


def test_page_workflow_state_mutation_stays_in_page_state_helper():
    allowed = {Path("app/models/page_state.py")}
    offenders: list[str] = []
    status_assignment_pattern = re.compile(r"\.\s*status\s*=\s*PageStatus\.")
    workflow_attr_assignment_pattern = re.compile(r"\.\s*(error_message|ocr_invalidated_reason)\s*=")
    for path in sorted(APP_DIR.rglob("*.py")):
        if path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        if status_assignment_pattern.search(source):
            offenders.append(f"{path}: PageStatus assignment")
        for match in workflow_attr_assignment_pattern.finditer(source):
            offenders.append(f"{path}: {match.group(0)}")
    assert offenders == []


def test_proof_signal_contract_does_not_restore_proof_saved():
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "proof_saved" in source:
            offenders.append(str(path))
    assert offenders == []


def test_proof_ui_does_not_create_unscoped_text_or_status_changes():
    offenders: list[str] = []
    pattern = re.compile(r"ProofChangeSet\([^)]*(text_changed|status_changed)\s*=")
    for path in sorted(PROOF_UI_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            line_no = source.count("\n", 0, match.start()) + 1
            offenders.append(f"{path}:{line_no}: {match.group(0)}")
    assert offenders == []


def test_proof_change_contract_names_scope_requirement():
    source = Path("app/core/proof_change.py").read_text(encoding="utf-8")
    assert "def requires_line_scope" in source
    assert "def has_required_scope" in source
    persist_source = Path("app/services/proof_persistence_service.py").read_text(encoding="utf-8")
    assert "change.has_required_scope" in persist_source


def test_layout_panel_user_edits_go_through_layout_edit_service():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    edit_service_source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    viewer_source = Path("app/ui/widgets/image_viewer.py").read_text(encoding="utf-8")

    assert "LayoutEditService" in layout_source
    assert "LayoutEditCommand" in layout_source
    assert "self._layout_edit_service.apply(" in layout_source
    assert "block_geometry_change_requested" in viewer_source
    assert "self._viewer.block_geometry_change_requested.connect(" in layout_source
    assert "self._viewer.block_moved.connect(" not in layout_source
    for retired_helper in (
        "def _apply_subtype_to_block",
        "def _merge_blocks_into_bbox",
        "def _update_existing_manual_binding_bbox",
        "def _bind_manual_block_to_paddle",
        "def _preserve_inline_formula_origin_binding",
        "def _mark_generated_inline_formula_handled",
    ):
        assert retired_helper not in layout_source
    for direct_service_call in (
        "self._layout_edit_service.create_block(",
        "self._layout_edit_service.delete_block(",
        "self._layout_edit_service.change_block_kind(",
        "self._layout_edit_service.merge_blocks_into_bbox(",
        "self._layout_edit_service.persist_user_block_geometry(",
    ):
        assert direct_service_call not in layout_source
    for direct_blocks_write in (
        ".blocks =",
        ".blocks.append(",
        ".blocks.remove(",
        ".blocks.pop(",
        ".blocks.clear(",
    ):
        assert direct_blocks_write not in layout_source
    assert "class LayoutEditCommand" in edit_service_source
    assert "def apply(" in edit_service_source
    assert "restore_blocks" in edit_service_source
    for internal_mutation in (
        "def _create_block(",
        "def _delete_block(",
        "def _change_block_kind(",
        "def _merge_blocks_into_bbox(",
        "def _persist_user_block_geometry(",
    ):
        assert internal_mutation in edit_service_source


def test_paddle_binding_is_typed_state_not_app_payload_write_path():
    block_state_source = Path("app/models/block_state.py").read_text(encoding="utf-8")
    artifact_source = Path("app/core/paddle_artifact_index.py").read_text(encoding="utf-8")
    edit_service_source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "paddle_binding_dict(block" in block_state_source
    assert "set_paddle_binding(block" in block_state_source
    assert "APP_PAYLOAD_KEYS" not in block_state_source
    assert "PADDLE_BINDING_KEY" not in artifact_source
    assert "PADDLE_BINDING_KEY" not in layout_source
    assert "set_paddle_binding(block" in artifact_source
    assert "set_paddle_binding(block" in edit_service_source
    assert "set_paddle_binding(block" not in layout_source
    assert "PaddleArtifactIndex" in edit_service_source
    assert "PaddleArtifactIndex" not in layout_source
    assert "set_paddle_binding(block" in hanwang_source
    assert "paddle_binding_json" in store_source
    assert "PaddleBinding.from_dict" in store_source
    assert "app_payload.pop(PADDLE_BINDING_KEY" not in store_source
    assert "app_payload[PADDLE_BINDING_KEY]" not in hanwang_source
    assert "set_payload_entries(block, {\n        PADDLE_BINDING_KEY" not in hanwang_source


def test_paddle_raw_label_fields_are_not_app_payload_state():
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    for key in ("PADDLE_BLOCK_LABEL_KEY", "PADDLE_BLOCK_BBOX_KEY"):
        assert f"app_payload.pop({key}" not in store_source


def test_block_ocr_invalidation_is_typed_state_not_app_payload_write_path():
    block_state_source = Path("app/models/block_state.py").read_text(encoding="utf-8")
    edit_service_source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    ocr_source = Path("app/services/ocr_pipeline.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "OCR_TEXT_INVALIDATED_KEY" not in block_state_source
    assert "OCR_INVALIDATION_KIND_KEY" not in block_state_source
    assert "setattr(block, \"ocr_invalidated_reason\"" in block_state_source
    assert "mark_ocr_text_invalidated(block" in edit_service_source
    assert "mark_ocr_text_invalidated(block" not in layout_source
    assert "is_ocr_text_invalidated(block)" in layout_source
    assert "is_ocr_text_invalidated(block)" in ocr_source
    assert "is_ocr_text_invalidated(block)" in hanwang_source
    assert "ocr_invalidated_reason" in store_source
    assert "app_payload[OCR_TEXT_INVALIDATED_KEY]" not in layout_source
    assert "set_payload_entries(block, {\n            OCR_TEXT_INVALIDATED_KEY" not in layout_source


def test_manual_layout_merge_details_are_events_not_app_payload_state():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "MANUAL_MERGE_FROM_KEY" not in layout_source
    assert "MANUAL_DRAW_BBOX_KEY" not in layout_source
    assert "app_payload.pop(MANUAL_MERGE_FROM_KEY" not in store_source
    assert "app_payload.pop(MANUAL_DRAW_BBOX_KEY" not in store_source


def test_generated_inline_formula_anchor_is_block_origin_not_app_payload_state():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    overlay_service_source = Path("app/services/layout_overlay_service.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    inline_state_source = Path("app/core/inline_formula_edit_state.py").read_text(encoding="utf-8")

    for key in (
        "UI_GENERATED_INLINE_FORMULA_BLOCK_KEY",
        "UI_INLINE_FORMULA_ORIGIN_BBOX_KEY",
        "UI_INLINE_FORMULA_PARENT_LABEL_KEY",
    ):
        assert key not in layout_source
        assert f"app_payload.pop({key}" not in store_source
    assert "origin=BlockOrigin(" not in layout_source
    assert "origin=BlockOrigin(" in overlay_service_source
    assert "original_bbox=overlay.bbox" in overlay_service_source
    assert "inline_formula_origin_bbox(block)" in overlay_service_source
    assert 'get("type")' not in inline_state_source


def test_layout_panel_does_not_parse_raw_layout_artifacts_directly():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    overlay_service_source = Path("app/services/layout_overlay_service.py").read_text(encoding="utf-8")
    index_source = Path("app/core/paddle_artifact_index.py").read_text(encoding="utf-8")
    normalized_source = Path("app/core/normalized_layout_artifact.py").read_text(encoding="utf-8")

    assert "LayoutOverlayService" in layout_source
    for forbidden in (
        "raw_layout_records",
        "raw_block_payload",
        "ROUTE_SUBBLOCKS_FIELD",
        "formula_texts_by_subblock_bbox",
        "line_routes_for_block",
        "bbox_from_variant",
    ):
        assert forbidden not in layout_source
    assert "normalized_layout_regions(page)" in overlay_service_source
    assert "raw_layout_records" not in overlay_service_source
    assert "ROUTE_SUBBLOCKS_FIELD" not in overlay_service_source
    assert "line_routes_for_block" not in overlay_service_source
    assert "routing_plan_for_block_record" in overlay_service_source
    assert "normalized_layout_regions(page)" in index_source
    assert "raw_layout_records" not in index_source
    assert "class NormalizedLayoutArtifact" in normalized_source
    assert "class LayoutRegion" in normalized_source
    assert "class LayoutSubregion" in normalized_source


def test_api_layout_blocks_are_projected_from_layout_snapshot():
    source = Path("app/core/layout_analyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(source, node) or ""

    api_source = functions["_extract_api_blocks"]
    assert "def _append_api_block" not in source
    assert "layout_snapshot_from_normalized_artifact(" in api_source
    assert "adopt_page_layout_snapshot(" in api_source
    assert "Block(" not in api_source


def test_layout_snapshot_contract_lives_in_model_layer():
    service_source = Path("app/services/layout_snapshot.py").read_text(encoding="utf-8")
    model_source = Path("app/models/layout_snapshot.py").read_text(encoding="utf-8")
    projection_source = Path("app/models/layout_snapshot_projection.py").read_text(encoding="utf-8")

    assert "class LayoutSnapshot" not in service_source
    assert "class LayoutBlockSnapshot" not in service_source
    assert "class LayoutSnapshot" in model_source
    assert "class LayoutBlockSnapshot" in model_source
    assert "from app.models.layout_snapshot import" in service_source
    assert "def sync_page_layout_snapshot_from_projection" in projection_source
    assert "def adopt_page_layout_snapshot" in projection_source


def test_project_store_persists_layout_snapshot_without_service_dependency():
    source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    assert "from app.models.layout_snapshot_projection import" in source
    assert "CREATE TABLE IF NOT EXISTS layout_snapshot" in source
    assert "def _save_layout_snapshot" in source
    assert "def _load_layout_snapshot" in source
    assert "layout_snapshot_for_page(" in source
    assert "set_layout_snapshot_for_page(" in source
    assert "sync_page_layout_snapshot_from_projection(" in source
    assert "from app.services.layout_snapshot" not in source


def test_layout_edit_service_records_snapshot_edits_without_projection_backflow():
    source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    assert "def record_edit" not in source
    assert "sync_snapshot" not in source
    assert "sync_page_layout_snapshot_from_projection" not in source

    record_source = _function_source(source, "record_snapshot_edit")
    assert "LayoutEditEvent(" in record_source
    assert "page.layout_edit_events.append(event)" in record_source
    assert "source_engine=" not in record_source


def test_layout_edit_service_reads_ocr_lines_from_observation_store():
    source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")

    assert "block_ocr_line_observations" in source
    assert re.search(r"(?<!clear_)block_ocr_lines\(", source) is None


def test_truth_map_records_layout_snapshot_as_current_boundary():
    source = Path("CURRENT_TRUTH_MAP.md").read_text(encoding="utf-8")

    assert "LayoutSnapshot`：当前采用的版面真值 contract" in source
    assert "SQLite `layout_snapshot` 表已作为 API 版面分析、人工编辑和项目保存/加载" in source
    assert "后续应抽 `PaddleArtifact`、`LayoutSnapshot`" not in source


def test_hanwang_text_slice_routing_reads_routing_plan():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(source, node) or ""

    text_route_source = functions["_text_route_bboxes_for_block"]
    assert "routing_plan_for_block_record(block, width, height).text_slices" in text_route_source
    assert "text_slice_routes_for_block(" not in text_route_source

    has_routes_source = functions["_has_route_subblocks"]
    assert "routing_plan_for_block_record(block, width, height).has_layout_routes" in has_routes_source
    assert "has_layout_line_routes(" not in has_routes_source

    assemble_routes_source = functions["_assemble_layout_route_lines"]
    assert "routing_plan_for_block_record(block, width, height).lines" in assemble_routes_source
    assert "line_routes_for_block(block" not in assemble_routes_source

    refine_routes_source = functions["_refine_layout_text_route_bands_from_image"]
    assert "routing_plan_for_block_record(block, width, height).lines" in refine_routes_source
    assert "line_routes_for_block(block" not in refine_routes_source
    assert "routing_line_to_record(route)" in refine_routes_source
    assert "block[LAYOUT_LINE_ROUTES_FIELD]" not in refine_routes_source
    apply_routes_source = functions["_apply_layout_line_route_records"]
    assert "ppvl_blocks[block_idx][LAYOUT_LINE_ROUTES_FIELD]" in apply_routes_source

    run_source = functions["run_micro_recblock"]
    assert "build_page_ocr_line_route_attachment(" in run_source
    assert "apply_page_ocr_line_route_attachment(" in run_source
    assert "attach_page_ocr_line_routes(" not in run_source
    assert "line_routes_for_block" not in source


def test_hanwang_recognize_uses_single_layout_ocr_input_plan():
    source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    assert "class _LayoutOcrInputPlan" in source
    recognize_source = _function_source(source, "recognize_page_blocks")
    assert "_compile_layout_ocr_input_plan(page)" in recognize_source
    assert "_current_layout_blocks_for_ocr(page)" not in recognize_source
    assert "_routed_manual_structure_blocks(page)" not in recognize_source


def test_layout_routing_service_uses_typed_producer_not_route_dict_apis():
    service_source = Path("app/services/layout_routing_plan.py").read_text(encoding="utf-8")
    producer_source = Path("app/core/paddle_line_routing.py").read_text(encoding="utf-8")

    assert "layout_routing_plan_for_block" in service_source
    assert "line_routes_for_block" not in service_source
    assert "text_slice_routes_for_block" not in service_source
    assert "has_layout_line_routes" not in service_source
    assert "def build_layout_routing_plan" in producer_source
    build_legacy_source = _function_source(producer_source, "build_layout_line_routes")
    assert "build_layout_routing_plan(block, width, height).lines" in build_legacy_source
    read_source = _function_source(producer_source, "layout_routing_plan_for_block")
    assert "block.pop(" not in read_source
    assert "block[LAYOUT_LINE_ROUTES_FIELD]" not in read_source
    attach_source = _function_source(producer_source, "attach_page_ocr_line_routes")
    assert "build_page_ocr_line_route_attachment(" in attach_source
    assert "apply_page_ocr_line_route_attachment(" in attach_source
    build_attachment_source = _function_source(producer_source, "build_page_ocr_line_route_attachment")
    assert "block.pop(" not in build_attachment_source
    assert "block[LAYOUT_LINE_ROUTES_FIELD]" not in build_attachment_source


def test_paddle_layout_schema_does_not_restore_markdown_text_alias():
    schema_source = Path("app/core/paddle_layout_schema.py").read_text(encoding="utf-8")

    assert "include_markdown" not in schema_source
    assert 'PADDLE_TEXT_KEYS = ("block_content",)' in schema_source


def test_deleted_inline_formula_state_is_layout_event_not_raw_mutation():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    edit_service_source = Path("app/services/layout_edit_service.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")

    assert "UI_DELETED_INLINE_FORMULA_KEY" not in layout_source
    assert "UI_DELETED_INLINE_FORMULA_KEY" not in hanwang_source
    assert "mark_inline_formula_origin_handled" in edit_service_source
    assert "mark_inline_formula_origin_handled" not in layout_source
    assert "filter_handled_inline_formula_subblocks" in hanwang_source


def test_hanwang_bbox_audit_is_typed_ocr_audit_not_app_payload_state():
    model_source = Path("app/models/project.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "ocr_audit:" in model_source
    assert "ocr_audit_json" in store_source
    assert "app_payload.pop(HANWANG_BBOX_AUDIT_KEY" not in store_source
    assert "app_payload[HANWANG_BBOX_AUDIT_KEY]" not in hanwang_source
    assert "app_payload.get(HANWANG_BBOX_AUDIT_KEY)" not in hanwang_source
    assert "ocr_audit=ocr_audit" in hanwang_source


def test_table_text_layer_cells_are_typed_state_not_app_payload_state():
    model_source = Path("app/models/project.py").read_text(encoding="utf-8")
    service_source = Path("app/services/table_text_layer_service.py").read_text(encoding="utf-8")
    ir_source = Path("app/export/ir_builder.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "table_text_layer_cells:" in model_source
    assert "table_text_layer_cells_json" in store_source
    assert "app_payload.pop(TABLE_TEXT_LAYER_CELLS_KEY" not in store_source
    assert "block.table_text_layer_cells = cells" in service_source
    assert "block.table_text_layer_cells" in ir_source
    assert "payload_get(block, TABLE_TEXT_LAYER_CELLS_KEY)" not in ir_source


def test_table_text_layer_service_reads_layout_from_snapshot_view():
    source = Path("app/services/table_text_layer_service.py").read_text(encoding="utf-8")

    assert "iter_page_layout_block_views" in source
    assert "from app.models.layout_projection import page_layout_blocks" not in source
    assert "view.bbox.to_dict()" in source
    assert "view.block_type != BlockType.TABLE" in source


def test_table_text_layer_service_reads_ocr_lines_from_observation_store():
    source = Path("app/services/table_text_layer_service.py").read_text(encoding="utf-8")

    assert "block_ocr_line_observations" in source
    assert "block_ocr_lines" not in source


def test_export_service_reads_ocr_lines_from_observation_store():
    source = Path("app/services/export_service.py").read_text(encoding="utf-8")

    assert "block_ocr_line_observations" in source
    assert "block_has_ocr_line_observations" in source
    assert "block_ocr_lines" not in source
    assert "block_has_ocr_lines" not in source


def test_proof_stats_service_reads_ocr_lines_from_observation_store():
    source = Path("app/services/proof_stats_service.py").read_text(encoding="utf-8")

    assert "iter_project_ocr_line_observation_occurrences" in source
    assert "iter_project_ocr_line_occurrences" not in source


def test_proof_line_utils_reads_ocr_lines_from_observation_store():
    source = Path("app/core/proof_line_utils.py").read_text(encoding="utf-8")

    assert "block_ocr_line_observations" in source
    assert "block_ocr_lines" not in source


def test_proof_readers_use_ocr_observation_store_not_block_lines_projection():
    targets = [
        Path("app/core/proof_line_facts.py"),
        Path("app/core/proof_occurrence.py"),
        Path("app/core/quality_probe.py"),
        Path("app/services/proof_crop_service.py"),
    ]

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "block_ocr_lines" not in source
        assert "block_has_ocr_lines" not in source

    assert "block_ocr_line_observations" in Path("app/core/proof_line_facts.py").read_text(encoding="utf-8")
    assert "block_ocr_line_observations" in Path("app/core/proof_occurrence.py").read_text(encoding="utf-8")
    assert "block_has_ocr_line_observations" in Path("app/core/quality_probe.py").read_text(encoding="utf-8")
    assert "block_ocr_line_observations" in Path("app/services/proof_crop_service.py").read_text(encoding="utf-8")


def test_project_store_line_table_does_not_restore_retired_proof_columns():
    source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    line_table = re.search(
        r"CREATE TABLE IF NOT EXISTS line \((.*?)\);",
        source,
        flags=re.DOTALL,
    )
    assert line_table is not None
    assert "_le" "gacy_line_table_proof_values" not in source
    assert "_migrate_legacy_line_proof_columns" not in source
    assert "original_text" not in line_table.group(1)
    assert "final_text" not in line_table.group(1)
    assert "final_text_set" not in line_table.group(1)
    assert "proof_status" not in line_table.group(1)
    assert "line.final_text, int(line.final_text_set)" not in source
    assert "line.proof_status.value, bb.x" not in source
    assert "UPDATE line SET text=?, final_text=?" not in source
    assert "INSERT INTO line (uid, block_id, text, final_text" not in source
    assert "ALTER TABLE line ADD COLUMN final_text" not in source
    assert "ALTER TABLE line ADD COLUMN final_text_set" not in source
    assert "ALTER TABLE line ADD COLUMN proof_status" not in source
    assert 'final_text=r["final_text"]' not in source
    assert 'proof_status=ProofStatus(r["proof_status"])' not in source


def test_llm_pre_review_is_not_restored():
    offenders: list[str] = []
    forbidden_patterns = (
        "LlmReviewStatus",
        "llm_suggestion",
        "llm_reason",
        "llm_review_status",
        "FakeLlmPreReviewEngine",
        "LlmCandidateProvider",
        "llm_rules",
    )
    for path in sorted(APP_DIR.rglob("*.py")):
        if path.name == "__pycache__":
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in source:
                offenders.append(f"{path}: {pattern}")
    assert offenders == []


def test_auto_flag_uses_service_not_retired_line_mutation_helper():
    source = Path("app/core/proof_status.py").read_text(encoding="utf-8")
    assert "def apply_auto_flag" not in source
    pipeline_source = Path("app/services/ocr_pipeline.py").read_text(encoding="utf-8")
    assert "apply_auto_flag" not in pipeline_source


def test_line_model_has_no_retired_final_text_mutation_wrapper():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    assert "def update_final_text" not in source
    assert "def ensure_text_contract" not in source
    assert "def __setattr__" not in source
    assert "original_text" not in source
    assert "fill_original" not in source
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Line":
            line_source = ast.get_source_segment(source, node) or ""
            assert "proof_state:" not in line_source
            assert "ProofLineState" not in line_source
            assert "if not self.ocr_text" not in line_source
            assert "self.ocr_text = self.text" not in line_source
            line_methods = {
                child.name for child in node.body
                if isinstance(child, ast.FunctionDef)
            }
            assert "to_dict" not in line_methods
            assert "set_proof_text" not in line_methods
            assert "set_proof_status" not in line_methods
            assert "apply_proof_state" not in line_methods
            break
    else:
        raise AssertionError("Line class not found")


def test_proof_state_store_does_not_consume_retired_line_attr():
    source = Path("app/models/proof_line_state_store.py").read_text(encoding="utf-8")
    assert "_consume_legacy_state_attr" not in source
    assert "_remove_legacy_state_attr" not in source
    assert "values.pop(\"proof_state\"" not in source
    assert "ProofLineState(line_uid=_line_uid(line))" in source


def test_quality_probe_app_config_does_not_restore_retired_ratio_key():
    config_source = Path("app/core/app_config.py").read_text(encoding="utf-8")
    probe_source = Path("app/core/quality_probe.py").read_text(encoding="utf-8")
    quality_dialog_source = Path("app/ui/widgets/quality_stats_dialog.py").read_text(encoding="utf-8")
    assert "quality_probe_target_ratio" not in config_source
    assert "TOPIC_PROBE_OBSERVED" not in probe_source
    assert "沿用旧比例" not in quality_dialog_source
    config_keys_source = _function_source(probe_source, "sampler_config_from_app_config")
    assert "quality_probe_target_ratio" not in config_keys_source
    mapping_source = probe_source.split("def sampler_config_from_app_config", 1)[0]
    assert "quality_probe_target_ratio" not in mapping_source


def test_block_model_does_not_reconstruct_origin_from_payloads():
    source = Path("app/models/project.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Block":
            block_source = ast.get_source_segment(source, node) or ""
            assert "_origin_from_legacy_fields" not in block_source
            assert "_sync_legacy_source_fields_from_origin" not in block_source
            assert "app_payload.get" not in block_source
            assert "raw_payload.get" not in block_source
            assert "_normalized_default_ocr_policy" not in block_source
            assert "_ocr_policy_label" not in block_source
            break
    else:
        raise AssertionError("Block class not found")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")
    assert "_origin_from_legacy_fields" not in store_source


def test_proof_line_facts_does_not_reconstruct_state_from_retired_mirrors():
    source = Path("app/core/proof_line_facts.py").read_text(encoding="utf-8")
    assert "ProofLineState(" not in source
    assert "final_text=str(getattr" not in source
    assert "proof_status=getattr" not in source


def test_proof_field_writes_stay_inside_approved_boundaries():
    allowed_files = {
        Path("app/core/project_store.py"),
        Path("app/core/ocr_proof_projection.py"),
        Path("app/services/proof_edit_service.py"),
        Path("app/services/proof_probe_text_service.py"),
    }
    proof_attrs = {"final_text", "final_text_set", "proof_status"}
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr not in proof_attrs:
                continue
            owner = ast.unparse(target.value)
            if "line" not in owner.lower() and owner != "self":
                continue
            if path not in allowed_files:
                offenders.append(f"{path}:{target.lineno}: {owner}.{target.attr}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "update_final_text":
                continue
            if path not in allowed_files:
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(func.value)}.update_final_text()")
    assert offenders == []


def test_application_code_uses_proof_mutation_helpers_for_line_proof_writes():
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if path == Path("app/models/project.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _assigned_attr_targets(tree):
            if target.attr not in {"final_text", "final_text_set", "proof_status"}:
                continue
            owner = ast.unparse(target.value)
            if "line" in owner.lower():
                offenders.append(f"{path}:{target.lineno}: {owner}.{target.attr}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "update_final_text":
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(func.value)}.update_final_text()")
            if func.attr in {
                "set_proof_text",
                "set_proof_status",
                "apply_proof_state",
            }:
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(func.value)}.{func.attr}()")
    assert offenders == []


def test_read_side_uses_proof_line_facts_adapter():
    target_files = [
        Path("app/export/ir_builder.py"),
        Path("app/core/quality_probe.py"),
        Path("app/services/export_service.py"),
        Path("app/services/char_index_service.py"),
        Path("app/services/proof_crop_service.py"),
        Path("app/services/proof_probe_text_service.py"),
        Path("app/services/proof_stats_service.py"),
        Path("app/core/paddle_line_routing.py"),
        Path("app/ui/recognize/ocr_panel.py"),
        Path("app/ui/recognize/layout_panel.py"),
    ]
    forbidden_attrs = {"display_text", "final_text", "proof_status", "ocr_text"}
    offenders: list[str] = []
    for path in target_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in forbidden_attrs:
                continue
            owner = ast.unparse(node.value)
            if owner == "line":
                offenders.append(f"{path}:{node.lineno}: .{node.attr}")
    assert offenders == []


def test_proof_ui_does_not_read_line_text_or_status_directly():
    target_files = [
        PROOF_UI_DIR / "h_proof.py",
        PROOF_UI_DIR / "v_proof.py",
    ]
    forbidden_attrs = {"display_text", "proof_status", "ocr_text"}
    offenders: list[str] = []
    for path in target_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in forbidden_attrs:
                continue
            owner = ast.unparse(node.value)
            if owner in {"line", "entry.line", "projection.line", "self._line"}:
                offenders.append(f"{path}:{node.lineno}: {owner}.{node.attr}")
    assert offenders == []


def test_hproof_uses_projection_as_single_line_runtime_fact():
    source = (PROOF_UI_DIR / "h_proof.py").read_text(encoding="utf-8")
    assert "self._items" not in source
    assert "self._units" not in source
    assert "self._line_view_models" not in source
    assert "model_item" not in source
    for retired_attr in (
        "self._pages",
        "self._projections",
        "self._current_idx",
        "self._selected_page_number",
        "self._show_formula_debug",
        "self._show_table_debug",
        "self._pending_external_requests",
        "self._filter_updating",
        "self._editable",
        "self._loaded_display_text",
        "self._loaded_line_signature",
        "self._external_conflict",
    ):
        assert retired_attr not in source
    assert "class _HProofSaveResult" not in source
    assert "ProofEditStatus" in source
    assert "ProofEditorRebuildState" in source
    assert "proof_rebuild_gate_for_editor_state" in source
    assert "HProofRuntimeSession" in source
    assert "HProofLineEditSession" in source
    assert "proof_request_matches_line" not in source
    assert "consume_external_refresh_plan" in source
    session_source = Path("app/services/proof_hproof_session.py").read_text(encoding="utf-8")
    assert "ProofExternalRefreshQueue" in session_source
    assert "ProofExternalRefreshPlan" in session_source
    assert "HProofExternalRefreshPlan" not in session_source
    assert "self._session.projections" in source
    assert "self._edit_session" in source

    projection_source = Path("app/core/proof_projection.py").read_text(encoding="utf-8")
    assert "model_item" not in projection_source

    session_source = Path("app/services/proof_hproof_session.py").read_text(encoding="utf-8")
    assert "HProofRuntimeSession" in session_source
    assert "HProofLineEditSession" in session_source
    assert "current_projection_index" in session_source


def test_vproof_uses_named_slots_and_shared_identity_helpers():
    source = (PROOF_UI_DIR / "v_proof.py").read_text(encoding="utf-8")
    assert "List[Tuple[Line, int, int, int]]" not in source
    assert "def _build_text_map" not in source
    for retired_attr in (
        "self._pages",
        "self._current_page_idx",
        "self._loaded_page_key",
        "self._loaded_text",
        "self._pending_external_lines",
        "self._pending_external_page_keys",
        "self._text_map",
        "self._reference_context",
        "self._entry_pos_by_key",
    ):
        assert retired_attr not in source
    assert "CharIndexService._page_key" not in source
    assert "proof_page_identity_key" in source
    assert "VProofOccurrenceSession" in source
    assert "allow_proof_rebuild" not in source
    assert "proof_rebuild_gate_for_reference_context" in source
    assert "pending_external_lines" not in source
    assert "pending_external_page_keys" not in source
    assert "queue_external_refresh" in source
    assert "consume_external_refresh_plan" in source
    occurrence_source = Path("app/services/proof_occurrence_session.py").read_text(encoding="utf-8")
    assert "ProofExternalRefreshQueue" in occurrence_source
    assert "ProofExternalRefreshPlan" in occurrence_source
    assert "VProofExternalRefreshPlan" not in occurrence_source

    reference_source = Path("app/services/proof_reference_context.py").read_text(encoding="utf-8")
    assert "ProofTextSlot" in reference_source
    assert "build_proof_reference_context" in reference_source
