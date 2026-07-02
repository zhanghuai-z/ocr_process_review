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
            assert "block.raw_payload" in fn_source
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
            assert 'app_payload.get("_layout_paddle_parent_index"' not in fn_source
            assert 'raw_payload.get("_layout_paddle_parent_index"' not in fn_source
            assert 'app_payload.get("block_label"' not in fn_source
            break
    else:
        raise AssertionError("_layout_row_from_block function not found")


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


def test_project_store_exposes_only_scoped_proof_line_write_port():
    from app.core.project_store import ProjectStore

    assert hasattr(ProjectStore, "update_proof_lines")
    assert not hasattr(ProjectStore, "update_line")
    assert not hasattr(ProjectStore, "update_line_with_chars")
    assert not hasattr(ProjectStore, "update_lines")


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
    assert "def __setattr__" not in source
    assert "original_text" not in source
    assert "fill_original" not in source
    tree = ast.parse(source, filename="app/models/project.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Line":
            line_methods = {
                child.name for child in node.body
                if isinstance(child, ast.FunctionDef)
            }
            assert "to_dict" not in line_methods
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
        Path("app/models/project.py"),
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


def test_application_code_uses_proof_state_methods_for_line_proof_writes():
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
            if not isinstance(func, ast.Attribute) or func.attr != "update_final_text":
                continue
            offenders.append(f"{path}:{node.lineno}: {ast.unparse(func.value)}.update_final_text()")
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
