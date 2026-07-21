# 测试迁移台账

状态基线：`a5768e0`（2026-07-20）。本台账记录 `7134bfc` 相对其父提交删除的测试文件，以及这些测试在 ProjectSession 架构中的去向。它不是旧测试的恢复清单：旧模型、旧 pipeline、旧 proof bus 和旧持久化入口不再是产品兼容契约。

## 证据范围

依据以下代码事实建立台账：

- `git diff --name-status 7134bfc^ 7134bfc -- tests` 显示 49 个删除文件/fixture，包含一个混合测试巨文件 `tests/test_core.py`。
- 同一提交新增了 `*_v2` 会话、布局、OCR、导出、项目文件和 proof 测试；这些文件是新架构的第一批可执行覆盖。
- 后续提交 `a2e0573`、`71cba38`、`7636cb2`、`8ba104b`、`b9e31cd`、`d02c41a`、`4f47263`、`9da2ea5`、`a5768e0` 分别补回了导入发布顺序、布局提交、布局工作台、proof 会话和 proof UI 交互。台账按当前代码和当前测试判断，不按 7134bfc 的提交说明推断覆盖。

分类含义如下：

- **退役旧模型兼容**：测试保护的是已删除的运行时对象树、旧字段、旧 bus、旧 pipeline、旧 store 或旧质量探针。不能为了恢复测试而恢复这些入口。
- **迁移有效用户行为**：行为仍属于产品契约，但原测试依赖旧边界；需要用 UID、不可变 record、snapshot、CAS 或当前 UI 会话重新表达。
- **已有新架构覆盖**：当前测试已经保护了同一项有效契约，旧文件不需要一比一恢复。
- **待补**：当前实现或测试没有可执行的等价覆盖；邻近测试只能作为上下文，不能标记为完成。

`test_core.py`、`test_proof_architecture.py` 和 `test_proof_hv_sync.py` 都是混合文件。下面按测试主题拆分它们；文件名会在其主导分类中出现一次，混合文件的有效行为和缺口在对应映射中单独说明。

## 可执行映射

以下映射使用当前工作树中的文件，均可直接执行：

| 映射 | 覆盖范围 | 命令 |
| --- | --- | --- |
| `S` | ProjectSession、项目文件、应用会话和严格 v2 store | `python -m pytest -q tests/test_application_session_v2.py tests/test_project_session.py tests/test_project_store_v2.py tests/test_project_file_service_v2.py` |
| `L` | 布局分析、snapshot、布局编辑和布局 UI | `python -m pytest -q tests/test_layout_analysis_session_v2.py tests/test_layout_edit_service_v2.py tests/test_layout_panel_session_v2.py` |
| `O` | CharOCR 输入、路由、物理行、调度和 OCR job | `python -m pytest -q tests/test_charocr_engine_v2.py tests/test_charocr_input_service_v2.py tests/test_ocr_dispatch_plan_v2.py tests/test_ocr_job_service_v2.py tests/test_layout_routing_plan.py tests/test_ppocr_layout_ownership.py tests/test_ppocr_mixed_text_routing.py tests/test_ppocr_route_compiler.py tests/test_ppocr_route_validation.py tests/test_ppocr_v6_prepass.py tests/test_physical_line_geometry.py tests/test_char_geometry_reconciler.py` |
| `E` | immutable export snapshot、proof/OCR 文本优先级、表格和格式导出 | `python -m pytest -q tests/test_export_snapshot_v2.py` |
| `U` | proof session、CAS、H/V proof UI 和质量统计 | `python -m pytest -q tests/test_proof_session_v2.py tests/test_proof_ui_session_v2.py` |
| `A` | 当前架构 ratchet，包括本任务新增的 UI 下层依赖 ratchet | `python -m pytest -q tests/architecture tests/test_architecture_import_ratchet.py` |
| `D` | 待补项的邻近回归，仅用于确认相邻边界没有回退，不代表缺口已覆盖 | `python -m pytest -q tests/test_ocr_job_service_v2.py tests/test_project_store_v2.py tests/test_proof_session_v2.py` |

## 退役旧模型兼容

这些删除文件的旧入口不应恢复。映射只说明仍然有效的部分由哪里承接，未列出的旧兼容语义没有保留义务。

| 删除文件 | 处理和当前映射 |
| --- | --- |
| `tests/charocr_native_route_fixture.py` | 旧 `Page`/原始 route fixture 退役；当前 fixture 为 `tests/charocr_routing_observation_fixture.py`，相关 typed route 回归执行 `O`。 |
| `tests/test_core.py` | 其中的旧模型投影、旧字段、旧 `ProjectStore`、旧 `OcrPipeline` 和旧 proof 写入口退役；保留的会话、布局、OCR、导出和 UI 主题分别由 `S`、`L`、`O`、`E`、`U` 覆盖。该巨文件本身不恢复。 |
| `tests/test_hanwang_proof_compat.py` | 旧 proof line/char 兼容层、旧 crop/index/service 入口退役；当前 proof record、alignment 和稳定 entry 行为由 `U` 的新契约覆盖。 |
| `tests/test_ocr_observation_views.py` | 旧 runtime projection drift 和 object-view adapter 退役；当前 active observation、export snapshot 和 OCR adoption 由 `E`、`O` 覆盖。 |
| `tests/test_proof_architecture.py` | 其中针对旧 payload、旧模型字段、旧 proof bus/service 的断言退役；当前边界断言由 `S`、`L`、`O`、`E`、`U` 和 `A` 分散承接。其未迁移的边界矩阵见“待补”。 |
| `tests/test_proof_services.py` | 旧 proof persistence/edit/crop/probe 服务入口退役；当前 aggregate CAS 和 proof UI service boundary 由 `U` 覆盖。 |
| `tests/test_quality_probe.py` | 旧 quality probe 运行时和配置入口退役，无兼容映射。 |
| `tests/test_quality_probe_blockers.py` | 旧 quality probe blocker 诊断入口退役，无兼容映射。 |
| `tests/test_quality_probe_round17.py` | 旧 round17 probe 脚本/数据契约退役，无兼容映射。 |
| `tests/test_quality_probe_round18.py` | 旧 round18 probe 脚本/数据契约退役，无兼容映射。 |
| `tests/test_workflow_controller_accessors.py` | 旧 controller 对 `OcrProject`/运行时对象树的 accessor 契约退役；当前 session、UID signal 和 shell workflow 由 `S` 覆盖。 |

## 迁移有效用户行为

以下行为仍有产品价值，但旧测试依赖已删除的对象或服务。当前映射是可运行的邻近覆盖；没有明确标为“已有新架构覆盖”的项目，不得据此宣称等价迁移完成。

| 删除文件 | 当前落点或迁移方向 | 邻近可执行映射 |
| --- | --- | --- |
| `tests/test_charocr_rotated_line.py` | native rotation 后的 line/char geometry 需要改写为 immutable OCR request/result 断言。 | `O` |
| `tests/test_charocr_route_artifacts.py` | route artifact 是非权威诊断输出，需改用 typed routing plan 和 session page UID。 | `O`、`D` |
| `tests/test_hanwang_large_line_input.py` | 大横向 physical line 的归一化和 batch 顺序仍是有效 OCR 行为。 | `O` |
| `tests/test_hanwang_latin_word_fallback.py` | Latin/engcut fallback、字符几何和 route ownership 需要在新 CharOCR result contract 下补齐。 | `O` |
| `tests/test_hproof_yaxis_quiet_load.py` | HProof 行布局、IME 输入和安静加载属于 UI 行为，需以 ProjectSession proof rows 重写。 | `U` |
| `tests/test_hproof_yaxis_verdicts.py` | proof confidence/verdict 的颜色和降级显示仍是用户行为，需绑定不可变 proof/OCR facts。 | `U` |
| `tests/test_image_viewer_frame.py` | layout block frame、边缘遮挡和 geometry edit 需要继续以 layout snapshot/UI projection 验证。 | `L` |
| `tests/test_layout_overlay_service.py` | overlay 计算应以 snapshot、artifact 和 stable UID 为输入，补 service/UI 合同测试。 | `L` |
| `tests/test_ocr_routing_observations.py` | changed layout、空响应、retry 和 page-local failure 仍是 OCR 可恢复行为。 | `O`、`D` |
| `tests/test_ppocr_route_execution_boundary.py` | linecut mask、typed route execution 和 skip policy 需要从旧 pipeline 入口迁移到 OcrJobService/engine boundary。 | `O` |
| `tests/test_proof_atom.py` | atom/slot/index 的稳定身份和 geometry 行为仍有效，需从旧 line projection 改为 proof alignment/observation records。 | `U` |
| `tests/test_proof_bbox_boxedit.py` | HProof bbox/inline editor 的输入和选择行为仍有效。 | `U` |
| `tests/test_proof_crosschar_batch.py` | 跨字符 gallery selection 和 batch apply 仍有效，需使用 immutable entry IDs。 | `U` |
| `tests/test_proof_direct_input_closure.py` | HProof editor focus、输入框外观和 row focus 仍有效。 | `U` |
| `tests/test_proof_external_refresh.py` | 外部更新按 UID、批次和 dirty conflict 处理仍有效，不能恢复旧全局 refresh queue。 | `U`、`D` |
| `tests/test_proof_hv_sync.py` | H/V proof 的用户可见同步仍有效；旧 global bus 实现退役，需基于 session refresh/CAS 重新覆盖。 | `U`、`D` |
| `tests/test_proof_interaction_slots.py` | slot editor 的删除、粘贴、替换和 fixed-mode 行为仍有效。 | `U` |
| `tests/test_proof_occurrence_session.py` | occurrence gallery 的当前页/offscreen page refresh 仍有效，旧 occurrence service 入口不保留。 | `U`、`D` |
| `tests/test_proof_rebuild_gate.py` | dirty/conflict/readonly 的 rebuild gate 仍有效，需成为 proof aggregate 状态而非 UI 私有镜像。 | `U` |
| `tests/test_proof_slot_residual.py` | 短文本补空格、长文本降级和 fixed editor 行为仍有效。 | `U` |
| `tests/test_proof_vproof_gallery_height_budget.py` | VProof gallery 高度预算和裁切仍是 UI 行为。 | `U` |
| `tests/test_proof_vproof_gallery_rendering.py` | VProof occurrence gallery 渲染仍是 UI 行为。 | `U` |
| `tests/test_proof_vproof_ime_persist_visibility.py` | VProof IME、持久化和可见性行为需要以 proof CAS 提交验证。 | `U` |
| `tests/test_proof_vproof_window_balance.py` | VProof 窗口和 gallery 的尺寸平衡仍是 UI 行为。 | `U` |
| `tests/test_vproof_gallery_crop_policy.py` | occurrence crop 的 geometry policy 仍有效，输入应改为 snapshot/OCR record。 | `U`、`D` |

## 已有新架构覆盖

以下旧测试的有效契约已经有当前 v2 或 typed boundary 测试保护，旧文件不应恢复：

| 删除文件 | 当前测试文件映射 |
| --- | --- |
| `tests/test_export_ocr_boundary.py` | `tests/test_export_snapshot_v2.py`：proof text 优先于 OCR text、snapshot-only export。 |
| `tests/test_hanwang_layout_snapshot_views.py` | `tests/test_layout_analysis_session_v2.py`、`tests/test_layout_panel_session_v2.py`：layout snapshot 是当前事实，UI 读取 session projection。 |
| `tests/test_layout_edit_service.py` | `tests/test_layout_edit_service_v2.py`：UID、CAS、immutable snapshot edit。 |
| `tests/test_layout_snapshot.py` | `tests/test_layout_analysis_session_v2.py`、`tests/test_layout_edit_service_v2.py`。 |
| `tests/test_normalized_layout_artifact.py` | `tests/test_layout_analysis_session_v2.py`：artifact provenance、adoption 和 revision。 |
| `tests/test_ocr_dispatch_plan.py` | `tests/test_ocr_dispatch_plan_v2.py`：page-scoped immutable dispatch facts。 |
| `tests/test_project_file_service.py` | `tests/test_project_file_service_v2.py`：严格 v2 store、asset staging、save-as、rollback；旧 live-model binding 不保留。 |
| `tests/test_proof_edit_service.py` | `tests/test_proof_session_v2.py`、`tests/test_proof_ui_session_v2.py`：aggregate CAS 和 UI service write。 |
| `tests/test_proof_hproof_session.py` | `tests/test_proof_session_v2.py`、`tests/test_proof_ui_session_v2.py`：session selection、dirty conflict 和 proof record。 |
| `tests/test_proof_vproof_direct_overwrite.py` | `tests/test_proof_ui_session_v2.py`：VProof 通过 service/CAS 写 proof，不改 OCR observation。 |
| `tests/test_table_text_layer_prepass.py` | `tests/test_export_snapshot_v2.py`、`tests/test_project_store_v2.py`：table text 是 typed repository/export snapshot 数据。 |

## 待补

当前没有可执行的等价测试文件，以下项目需要后续单独建 v2 测试；`D` 只能运行邻近回归，不能关闭缺口：

| 删除文件 | 缺口 | 建议的新测试文件 |
| --- | --- | --- |
| `tests/test_ocr_routing_run_audit_store.py` | `OcrRoutingRunAudit` model/service 仍存在，但当前 v2 `ProjectStore` 没有对应的持久化/恢复回归。 | `tests/test_ocr_routing_audit_v2.py` |
| `tests/test_project_diagnostics.py` | 旧 project diagnostics service 已退役；当前只有 route artifact/IR diagnostics，缺少 session-scoped diagnostics workflow。 | `tests/test_project_diagnostics_v2.py` |

混合删除文件还有以下未闭合部分：

- `tests/test_core.py` 中旧 export worker、旧 MainWindow lifecycle、旧质量探针和旧 persistence repair 片段没有被 `S/L/O/E/U` 等价覆盖；它们要么属于退役入口，要么需要拆成明确的 v2 用户行为测试。
- `tests/test_proof_architecture.py` 中的完整 boundary matrix 没有由一个新文件取代；当前 `tests/test_proof_session_v2.py`、`tests/test_proof_ui_session_v2.py` 和 `tests/architecture/test_ui_dependency_ratchet.py` 只覆盖已落地的边界。
- `tests/test_proof_hv_sync.py` 中旧 bus 的实现断言全部退役，但 cross-pane refresh、IME、gallery 和 conflict 的有效行为仍归入“迁移有效用户行为”。

## 完整性核对

按分类计数：退役旧模型兼容 11 个、迁移有效用户行为 25 个、已有新架构覆盖 11 个、待补 2 个，共 49 个删除文件。每个删除路径都保留在上面唯一一个主分类中；混合文件的主题拆分在“待补”中注明。当前任务只新增本台账和 architecture ratchet，不修改 `app` 生产代码，也不恢复任何旧兼容路径。
