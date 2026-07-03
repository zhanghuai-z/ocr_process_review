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


def test_retired_block_payload_helper_is_not_restored():
    assert not Path("app/core/block_payload.py").exists()


def test_ocr_dispatch_policy_uses_block_attributes_not_payload_guessing():
    source = Path("app/core/ocr_dispatch_policy.py").read_text(encoding="utf-8")
    assert "block_attributes(" in source
    assert "app_payload" not in source
    assert "raw_payload" not in source
    assert "block.source_label" not in source
    assert "PADDLE_BINDING_KEY" not in source


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
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_layout_row_from_block":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "route_source_label(block)" in fn_source
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


def test_paddle_binding_is_typed_state_not_app_payload_write_path():
    block_state_source = Path("app/models/block_state.py").read_text(encoding="utf-8")
    artifact_source = Path("app/core/paddle_artifact_index.py").read_text(encoding="utf-8")
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "paddle_binding_dict(block" in block_state_source
    assert "set_paddle_binding(block" in block_state_source
    assert "APP_PAYLOAD_KEYS" not in block_state_source
    assert "PADDLE_BINDING_KEY" not in artifact_source
    assert "PADDLE_BINDING_KEY" not in layout_source
    assert "set_paddle_binding(block" in artifact_source
    assert "set_paddle_binding(block" in layout_source
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
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    ocr_source = Path("app/services/ocr_pipeline.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    assert "OCR_TEXT_INVALIDATED_KEY" not in block_state_source
    assert "OCR_INVALIDATION_KIND_KEY" not in block_state_source
    assert "setattr(block, \"ocr_invalidated_reason\"" in block_state_source
    assert "mark_ocr_text_invalidated(block" in layout_source
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
    store_source = Path("app/core/project_store.py").read_text(encoding="utf-8")

    for key in (
        "UI_GENERATED_INLINE_FORMULA_BLOCK_KEY",
        "UI_INLINE_FORMULA_ORIGIN_BBOX_KEY",
        "UI_INLINE_FORMULA_PARENT_LABEL_KEY",
    ):
        assert key not in layout_source
        assert f"app_payload.pop({key}" not in store_source
    assert "origin=BlockOrigin(" in layout_source
    assert "original_bbox=bbox" in layout_source
    assert "_inline_formula_origin_bbox(block)" in layout_source


def test_deleted_inline_formula_state_is_layout_event_not_raw_mutation():
    layout_source = Path("app/ui/recognize/layout_panel.py").read_text(encoding="utf-8")
    hanwang_source = Path("app/engines/hanwang/micro_recblock.py").read_text(encoding="utf-8")

    assert "UI_DELETED_INLINE_FORMULA_KEY" not in layout_source
    assert "UI_DELETED_INLINE_FORMULA_KEY" not in hanwang_source
    assert "mark_inline_formula_origin_handled" in layout_source
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
    assert "HProofRuntimeSession" in source
    assert "HProofLineEditSession" in source
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

    reference_source = Path("app/services/proof_reference_context.py").read_text(encoding="utf-8")
    assert "ProofTextSlot" in reference_source
    assert "build_proof_reference_context" in reference_source
