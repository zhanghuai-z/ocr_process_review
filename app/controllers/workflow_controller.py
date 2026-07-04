"""工作流控制器：管理项目状态、步骤跳转、Worker 调度。

职责：
- 从 MainWindow 抽离项目创建/打开/保存逻辑
- 统一维护工作流状态机
- 根据状态启用/禁用步骤按钮
- 调度 LayoutWorker / OcrWorker
- 处理 OCR 完成后的状态流转
"""
from __future__ import annotations
import copy
import time
from threading import Lock
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.models.block_state import mark_ocr_text_invalidated
from app.core.logging import get_logger
from app.core.app_config import get_config
from app.core.proof_line_utils import iter_unique_page_text_lines
from app.core.proof_line_facts import proof_display_text
from app.core.project_store import ProjectStore
from app.core.proof_change import ProofChangeSet
from app.core.workflow_state import (
    STEP_HPROOF,
    STEP_IMPORT,
    STEP_LAYOUT,
    STEP_OCR,
    STEP_VPROOF,
    WorkflowProgressState,
    WorkflowViewState,
    compute_max_step,
    page_gate_info,
    pending_ocr_pages,
)
from app.core import quality_probe as qp
from app.engines.hanwang import native_cache
from app.engines.real_ocr_adapter import create_engine, get_engine_description
from app.models import (
    BBox, Block, BlockType, OcrProject, Page,
)
from app.models.layout_projection import page_layout_blocks, replace_page_layout_blocks
from app.models.ocr_character_observation import iter_line_ocr_char_occurrences, line_ocr_chars
from app.models.ocr_text_observation import line_ocr_review_flags
from app.models.ocr_observation import (
    iter_page_ocr_line_occurrences,
    line_ocr_bbox,
    page_has_ocr_result,
    project_all_pages_ocr_done,
    project_has_any_ocr_done_page,
    project_has_any_ocr_result,
    project_ocr_line_count,
)
from app.models.page_state import (
    invalidate_page_ocr,
    mark_page_imported,
    mark_page_layout_done,
    mark_page_layout_failed,
    mark_page_ocr_done,
    mark_page_ocr_failed,
    page_has_error,
    page_is_layout_analyzed,
    page_is_ocr_done,
    reconcile_page_ocr_done_from_result,
)
from app.services.ocr_pipeline import OcrPipeline
from app.services.ocr_dispatch_plan import count_text_ocr_blocks
from app.services.ocr_run_result import OcrProgress
from app.services.proof_auto_flag_service import ProofAutoFlagService
from app.services.proof_crop_service import ProofCropService
from app.services.proof_persistence_service import ProofPersistenceService

logger = get_logger(__name__)
PARALLEL_PROOF_PAGE_KEY_ATTR = "_parallel_proof_page_key"
OCR_PROGRESS_MIN_EMIT_INTERVAL_SECONDS = 0.08


class WorkflowController(QObject):
    """工作流控制器。"""

    # 状态变化信号
    project_changed = Signal(object)       # OcrProject
    step_enabled_changed = Signal(int)     # 允许的最大步骤
    step_requested = Signal(int)           # 请求跳转步骤
    layout_finished = Signal(object)       # List[Page]
    layout_progress = Signal(int, int)     # completed_index, total
    layout_stage = Signal(int, int, str)   # page_index, total, stage message
    ocr_finished = Signal(object)          # List[Page]
    ocr_progress = Signal(object)          # OcrProgress
    worker_error = Signal(str)             # 错误消息
    layout_cancelled = Signal()            # 版面分析被用户取消
    status_message = Signal(str)           # 状态栏消息
    # ── view-state ownership signals (本轮新增) ──
    current_step_changed = Signal(int)     # 当前激活的 step
    current_page_number_changed = Signal(int)  # 当前激活的 page_number
    layout_run_enabled_changed = Signal(bool)  # 顶部"运行版面"按钮可用性
    view_state_changed = Signal(object)    # WorkflowViewState
    progress_state_changed = Signal(object)  # WorkflowProgressState
    focus_page = Signal(int)               # page_number
    page_gate_state = Signal(int, str, bool, str, str)  # page_number, page_state, is_pending, reason_code, reason_text
    primary_action = Signal(int, str, str, bool)  # page_number, action_key, label, enabled

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Optional[OcrProject] = None
        self._store: Optional[ProjectStore] = None
        self._proof_auto_flag_service = ProofAutoFlagService()
        self._proof_crop_service = ProofCropService()
        self._max_step: int = STEP_IMPORT
        self._layout_worker = None
        self._ocr_worker = None
        self._proof_ocr_worker = None
        self._pending_layout_pages: Optional[List[Page]] = None
        self._pending_proof_pages: Optional[List[Page]] = None
        self._discard_parallel_proof_result: bool = False
        self._ocr_target_page_numbers: set[int] | None = None
        self._auto_start_ocr_after_layout = True
        self._queued_ocr_progress_callback: Optional[Callable] = None
        # proof 面板同步状态（供 sync_proof_panels 使用）
        self._proof_loaded_line_count: int = 0
        self._proof_loaded_signature: tuple = ()
        self._last_ocr_progress_completed_pages: int = 0
        self._hproof_panel = None
        self._vproof_panel = None
        # view-state ownership: 哪个 step 激活、哪个 page_number 激活、版面按钮可用性
        self._current_step: int = STEP_IMPORT
        self._current_page_number: int = 1
        self._layout_run_enabled: bool = False

    # ------------------------------------------------------------------ project management

    @property
    def project(self) -> Optional[OcrProject]:
        """当前 OcrProject（可能为 None）。供正确率统计等入口使用。"""
        return self._project

    @property
    def store(self) -> Optional[ProjectStore]:
        return self._store

    @property
    def max_step(self) -> int:
        return self._max_step

    def ensure_transient_project(self, name: str = "未命名项目") -> OcrProject:
        """确保存在一个临时项目对象，便于未保存状态下也能走通工作流。"""
        if self._project is None:
            self._project = OcrProject(name=name)
            self._sync_native_cache_dir()
            self.project_changed.emit(self._project)
        return self._project

    def get_open_step(self) -> int:
        """打开项目后的推荐落点。"""
        if self._max_step >= STEP_HPROOF:
            return STEP_HPROOF
        if self._max_step >= STEP_LAYOUT:
            return STEP_LAYOUT
        return STEP_IMPORT

    def get_text_ocr_block_count(self) -> int:
        if not self._project:
            return 0
        return count_text_ocr_blocks(self._project.pages)

    # ── 收口给 MainWindow 的窄 accessor ──────────────────────────────
    # 这些 property 是为了让 MainWindow 不必直接读 ``self._controller.project.*``
    # 拿 pages / 计算行数 / 判 analyzed-or-completed。语义和原 inline 读完全一致，
    # 仅多了一层 None-safe，便于未建项目时也能安全调用。

    @property
    def pages(self) -> List[Page]:
        """当前项目的 pages；无项目时返回空列表（不抛异常）。"""
        if self._project is None:
            return []
        return self._project.pages

    @property
    def has_pages(self) -> bool:
        return bool(self._project and self._project.pages)

    @property
    def total_line_count(self) -> int:
        """所有 page 的 total_lines 之和；无项目时返回 0。"""
        if self._project is None:
            return 0
        return project_ocr_line_count(self._project)

    @property
    def is_fully_analyzed(self) -> bool:
        """所有 page 都已完成版面分析；无 page 时返回 False。"""
        if not self._project or not self._project.pages:
            return False
        return all(page_is_layout_analyzed(page) for page in self._project.pages)

    @property
    def has_any_ocr_result(self) -> bool:
        """项目是否已有任意 OCR 结果；无项目时 False。"""
        if self._project is None:
            return False
        return project_has_any_ocr_result(self._project)

    @property
    def all_pages_ocr_done(self) -> bool:
        """所有页面是否都已完成 OCR；无项目时 False。"""
        if self._project is None:
            return False
        return project_all_pages_ocr_done(self._project)

    @property
    def cache_dir(self) -> "Path":
        """项目对应的缓存目录：``<.ocrproj 所在目录>/.cache``；
        临时项目（未保存）落在当前工作目录的 ``./.cache``。

        把 ``self._controller._project.db_path`` 这种私有访问收掉，避免
        MainWindow 触手伸进 controller 的实现细节。
        """
        from pathlib import Path as _Path
        db_path = self._project.db_path if self._project else None
        return _Path(db_path or ".").parent / ".cache"

    @property
    def charocr_cache_dir(self) -> "Path":
        return self.cache_dir / "hanwang_native"

    @property
    def active_charocr_cache_dir(self) -> "Path":
        self._sync_native_cache_dir()
        return native_cache.cache_dir()

    def _sync_native_cache_dir(self) -> None:
        native_cache.set_runtime_cache_dir(self.charocr_cache_dir)

    def clear_charocr_cache(self) -> "Path":
        self._sync_native_cache_dir()
        path = native_cache.clear_cache()
        self.status_message.emit(f"CharOCR 缓存已清空：{path}")
        return path

    def page_number_at(self, idx: int) -> Optional[int]:
        """按位置返回 page_number；越界或无项目时 None。"""
        if not self._project or not (0 <= idx < len(self._project.pages)):
            return None
        return self._project.pages[idx].page_number

    def page_by_number(self, page_number: int) -> Page | None:
        if not self._project:
            return None
        for page in self._project.pages:
            if page.page_number == page_number:
                return page
        return None

    def is_hanwang_mode(self) -> bool:
        return self._current_ocr_mode() == "hanwang"

    def ocr_engine_description(self) -> str:
        return get_engine_description(self._current_ocr_mode())

    # ── proof 面板同步：把"merge vs load"判定从 MainWindow 收回 ──────
    # 之前 MainWindow 自己 sum(total_lines) 决定 load 还是 merge，并维护
    # ``_proof_loaded_line_count``。现在所有权收到 controller，MainWindow
    # 只要在合适时机调 ``sync_proof_panels()``。

    def register_proof_panels(self, hproof, vproof) -> None:
        """注册两个校对面板，供 ``sync_proof_panels`` / ``refresh_proof_quality_probe_state`` 使用。"""
        self._hproof_panel = hproof
        self._vproof_panel = vproof

    @staticmethod
    def _bbox_signature(bbox: BBox | None) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        return int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)

    def _proof_pages_signature(self) -> tuple:
        if not self._project:
            return ()
        parts: list[tuple] = [("total_line_count", self.total_line_count)]
        for page_idx, page in enumerate(self._project.pages):
            parts.append((
                "page",
                page.uid,
                page.id,
                page_idx,
                page.page_number,
                page.display_image_path,
                page.width,
                page.height,
            ))
            for block, line, line_idx in iter_unique_page_text_lines(page):
                parts.append((
                    "line",
                    block.uid,
                    block.id,
                    block.order,
                    block.block_type.value,
                    line.uid,
                    line.id,
                    line_idx,
                    proof_display_text(line),
                    self._bbox_signature(line_ocr_bbox(line)),
                    line_ocr_review_flags(line),
                ))
                for char_occurrence in iter_line_ocr_char_occurrences(line):
                    char = char_occurrence.char
                    parts.append((
                        "char",
                        char.uid,
                        char_occurrence.char_index,
                        char.char,
                        self._bbox_signature(char.bbox),
                        char.bbox_source,
                        char.bbox_granularity,
                        char.token_text,
                        round(float(char.confidence), 6),
                    ))
        return tuple(parts)

    def sync_proof_panels(self, *, force_load: bool = False) -> None:
        """根据当前 project.pages 的 proof 数据同步两个校对面板。

        - ``force_load=True``：不论之前是否已 load 过，都走 load_pages（用于
          打开项目这种"全量初始化"场景）。
        - 否则按"已加载 → merge / 未加载 → load"切换，与原 MainWindow 实现等价。
        - 若 proof 数据签名没变化则什么都不做（避免 OCR 进度回调里频繁刷新）。
        """
        panels = (getattr(self, "_hproof_panel", None),
                  getattr(self, "_vproof_panel", None))
        if not all(panels) or not self.has_pages:
            return
        line_count = self.total_line_count
        if line_count <= 0:
            return
        signature = self._proof_pages_signature()
        if force_load:
            for p in panels:
                p.load_pages(self._project.pages)
            self._proof_loaded_line_count = line_count
            self._proof_loaded_signature = signature
            return
        if signature == getattr(self, "_proof_loaded_signature", ()):
            return
        if getattr(self, "_proof_loaded_line_count", 0) > 0:
            for p in panels:
                p.merge_pages(self._project.pages)
        else:
            for p in panels:
                p.load_pages(self._project.pages)
        self._proof_loaded_line_count = line_count
        self._proof_loaded_signature = signature

    def reset_proof_sync_state(self) -> None:
        """新建/打开项目前清掉 proof 同步计数。"""
        self._proof_loaded_line_count = 0
        self._proof_loaded_signature = ()

    def refresh_proof_quality_probe_state(self, step: int) -> None:
        """只刷新与 ``step`` 对应的那一个校对面板的 quality-probe 显示。

        与原 MainWindow 行为等价：进横校时只刷横校，进纵校时只刷纵校；
        ``step`` 不是 ``STEP_HPROOF/STEP_VPROOF`` 时静默 no-op。
        **不要**写成"两个面板都刷"——隐藏的 panel 一起 reload 会触发不必要的
        重排和样式重算（且不符合"行为零变化"承诺）。
        """
        if step == STEP_HPROOF:
            panel = getattr(self, "_hproof_panel", None)
        elif step == STEP_VPROOF:
            panel = getattr(self, "_vproof_panel", None)
        else:
            return
        if panel is None:
            return
        fn = getattr(panel, "refresh_quality_probe_state", None)
        if callable(fn):
            fn()

    # ── view-state accessors (current_step / current_page_number / layout_run_enabled) ──
    # 这些 ownership 之前散在 MainWindow（self._current_step / self._current_page_number /
    # 5 个手动 self._top_nav.set_layout_run_enabled(...) 调用），导致：
    #   - 谁是 step 真值不清（MainWindow vs controller.max_step）
    #   - prev/next step 计算用 MainWindow 局部缓存
    #   - layout 按钮可用性散在 5 个 worker 生命周期点
    # 现在 controller 统一持有 + 通过 signal 外播，MainWindow 退化为纯订阅者。

    @property
    def current_step(self) -> int:
        return self._current_step

    def workflow_view_state(self) -> WorkflowViewState:
        return WorkflowViewState(
            max_step=self._max_step,
            current_step=self._current_step,
            current_page_number=self._current_page_number,
            layout_run_enabled=self._layout_run_enabled,
            has_project=self._project is not None,
            total_pages=len(self._project.pages) if self._project else 0,
            total_lines=self.total_line_count,
        )

    def _emit_view_state(self) -> None:
        self.view_state_changed.emit(self.workflow_view_state())

    def set_current_step(self, step: int) -> None:
        """更新当前 step；变化时 emit signal。MainWindow 侧通过 signal 同步 UI。"""
        if step == self._current_step:
            return
        self._current_step = step
        self.current_step_changed.emit(step)
        self._emit_view_state()

    @property
    def current_page_number(self) -> int:
        return self._current_page_number

    def set_current_page_number(self, page_number: int) -> None:
        """更新当前激活的 page_number；变化时 emit signal。"""
        if page_number == self._current_page_number:
            return
        self._current_page_number = page_number
        self.current_page_number_changed.emit(page_number)
        self._emit_view_state()

    @property
    def layout_run_enabled(self) -> bool:
        return self._layout_run_enabled

    def set_layout_run_enabled(self, enabled: bool) -> None:
        """更新版面运行按钮可用性；变化时 emit signal。"""
        if enabled == self._layout_run_enabled:
            return
        self._layout_run_enabled = enabled
        self.layout_run_enabled_changed.emit(enabled)
        self._emit_view_state()

    def open_project(self, db_path: str) -> bool:
        """打开项目。"""
        try:
            if self._store:
                self._store.close()
            self._store = ProjectStore(db_path)
            self._store.open()
            self._project = self._store.load_project(project_id=1)
            if not self._project:
                self.worker_error.emit("项目文件无效或为空")
                return False
            self._sync_native_cache_dir()
            proof_stats = self._proof_crop_service.normalize_project(self._project)

            # 切换项目：先清空全局评测状态，再尝试从 sidecar 恢复
            qp.reset_active_store()
            side = qp.sidecar_path_for_project(self._project.db_path)
            if side:
                loaded = qp.load_store_from_path(side)
                if loaded is not None and len(loaded) > 0:
                    qp.set_active_store(loaded)
                    logger.info("Loaded quality probe sidecar: %d probes", len(loaded))

            self._max_step = self._compute_max_step()
            self.project_changed.emit(self._project)
            self.step_enabled_changed.emit(self._max_step)
            self._emit_view_state()
            fallback_warning = self._proof_fallback_warning(proof_stats, self._project.pages)
            if fallback_warning:
                self.status_message.emit(f"已打开：{db_path}；{fallback_warning}")
            else:
                self.status_message.emit(f"已打开：{db_path}")
            return True
        except Exception as e:
            logger.error("Failed to open project: %s", e)
            self.worker_error.emit(f"打开项目失败：{e}")
            return False

    def save_project_as(self, db_path: str) -> bool:
        """Save the current in-memory project to a project database path."""
        if not self._project:
            return False
        try:
            if self._store:
                self._store.close()
            self._store = ProjectStore(db_path)
            self._store.open()
            self._project.db_path = db_path
            self._project = self._store.save_project(self._project)
            self._save_quality_probe_sidecar()
            self._sync_native_cache_dir()
            self.project_changed.emit(self._project)
            self.status_message.emit(f"项目已保存：{db_path}")
            return True
        except Exception as e:
            logger.error("Save as failed: %s", e)
            self.worker_error.emit(f"保存失败：{e}")
            return False

    def save_project(self) -> bool:
        """保存项目。"""
        if not self._project or not self._store:
            return False
        try:
            self._store.save_project(self._project)
            self._save_quality_probe_sidecar()
            self.status_message.emit("项目已保存")
            return True
        except Exception as e:
            logger.error("Save failed: %s", e)
            self.worker_error.emit(f"保存失败：{e}")
            return False

    def _save_quality_probe_sidecar(self) -> bool:
        """Persist active quality-probe sidecar for the current project path."""
        if not self._project or not self._store:
            return False
        return ProofPersistenceService(self._store, self._project).persist_quality_probe_sidecar()

    def has_running_workers(self) -> bool:
        """是否存在仍在运行的后台任务。"""
        return any(
            self._worker_is_running(worker)
            for worker in (
                self._layout_worker,
                self._ocr_worker,
                self._proof_ocr_worker,
            )
        )

    def cancel_layout_analysis(self, *, wait_ms: int = 200, force: bool = False) -> bool:
        """Request cancellation for the active layout worker."""
        worker = self._layout_worker
        if not self._worker_is_running(worker):
            return False
        self._discard_parallel_proof_result = True
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        stopped = self._request_worker_stop(worker, wait_ms=wait_ms, force=force)
        if stopped:
            self._clear_worker_ref("_layout_worker", worker)
            self._on_layout_cancelled()
        else:
            self.status_message.emit("正在停止版面分析…")
        return True

    def cancel_running_workers(self, *, wait_ms: int = 1200, force: bool = False) -> bool:
        """Stop all background workers. Used by close/force-close paths."""
        requested = False
        all_stopped = True
        for attr_name in ("_layout_worker", "_ocr_worker", "_proof_ocr_worker"):
            worker = getattr(self, attr_name, None)
            if not self._worker_is_running(worker):
                continue
            requested = True
            stopped = self._request_worker_stop(worker, wait_ms=wait_ms, force=force)
            all_stopped = all_stopped and stopped
            if stopped:
                self._clear_worker_ref(attr_name, worker)
        if requested:
            self._discard_parallel_proof_result = True
            self._pending_layout_pages = None
            self._pending_proof_pages = None
            self._ocr_target_page_numbers = None
            self.set_layout_run_enabled(True)
            self.status_message.emit("后台任务已停止" if all_stopped else "后台任务正在停止…")
        return all_stopped

    def close_project(self) -> bool:
        """关闭当前项目，但不退出应用。"""
        if self.has_running_workers():
            self.status_message.emit("后台任务仍在运行，请等待完成后再关闭项目")
            return False
        if self._store:
            self._store.close()
        self._store = None
        self._project = None
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        self._discard_parallel_proof_result = False
        self._ocr_target_page_numbers = None
        self._auto_start_ocr_after_layout = True
        self._queued_ocr_progress_callback = None
        native_cache.reset_runtime_cache_dir()
        self.reset_proof_sync_state()
        self._max_step = STEP_IMPORT
        self._current_step = STEP_IMPORT
        self._current_page_number = 1
        self._layout_run_enabled = False
        qp.reset_active_store()
        self.project_changed.emit(None)
        self.step_enabled_changed.emit(self._max_step)
        self.current_step_changed.emit(self._current_step)
        self.current_page_number_changed.emit(self._current_page_number)
        self.layout_run_enabled_changed.emit(self._layout_run_enabled)
        self._emit_view_state()
        self.status_message.emit("项目已关闭")
        return True

    def auto_save(self, change: ProofChangeSet | None = None) -> None:
        """Persist proof mutations requested by proof panels."""
        if not self._project or not self._store:
            return
        try:
            ProofPersistenceService(self._store, self._project).persist(change)
        except Exception as e:
            logger.warning("Auto-save failed: %s", e)

    def ensure_quality_probe_sampled(self) -> bool:
        """采样触发入口 —— 进入校对步骤时由 UI 调用。

        触发规则：
        1. 当前已有 active store → 不重采（用户/上次会话已经在用）。
        2. AppConfig.quality_probe_auto_enable 为 False → 不自动触发。
        3. 项目未完成 OCR → 不触发（无内容可投放）。
        4. 否则按 SamplerConfig.from_app_config() 采样并设为 active。

        返回 True 表示当前已存在可用的评测 store（无论本次是否新采）。
        """
        if not self._project:
            return False
        if qp.get_active_store() is not None:
            return True
        # 是否启用自动评测
        try:
            from app.core.app_config import AppConfig
            auto = AppConfig.instance().get("quality_probe_auto_enable", True)
            # QSettings 可能把 bool 存成字符串
            if isinstance(auto, str):
                auto = auto.strip().lower() not in ("0", "false", "no", "")
            if not auto:
                return False
        except Exception:
            pass
        if not project_all_pages_ocr_done(self._project):
            return False
        try:
            cfg = qp.sampler_config_from_app_config()
            store = qp.ProbeSampler(cfg).sample(self._project)
            if len(store) == 0:
                return False
            qp.set_active_store(store)
            logger.info("Auto-sampled %d quality probes", len(store))
            # 立即落盘 sidecar，避免崩溃丢失采样结果
            if self._project.db_path:
                side = qp.sidecar_path_for_project(self._project.db_path)
                if side:
                    qp.save_store_to_path(store, side)
            return True
        except Exception as e:
            logger.warning("Quality probe sampling failed: %s", e)
            return False

    def close(self) -> None:
        if self._store:
            self._store.close()
        native_cache.reset_runtime_cache_dir()
        # 进程退出：清空全局评测状态
        qp.reset_active_store()

    # ------------------------------------------------------------------ step management

    def can_enter_step(self, step: int) -> bool:
        """检查是否允许进入某步骤。"""
        return step <= self._max_step

    def request_step(self, step: int) -> None:
        """请求跳转到某步骤（由 UI 触发）。"""
        if self.is_hanwang_mode() and step == STEP_OCR:
            self.handle_ocr_entry_requested("step_nav", self._current_page_number)
            return
        if not self.can_enter_step(step):
            self.status_message.emit("当前状态不允许进入该步骤")
            return
        self.step_requested.emit(step)

    def _compute_max_step(self) -> int:
        """根据项目状态计算可进入的最大步骤。"""
        return compute_max_step(self._project)

    def _update_max_step(self) -> None:
        """更新最大可进入步骤并通知 UI。"""
        self._max_step = self._compute_max_step()
        self.step_enabled_changed.emit(self._max_step)
        self._emit_view_state()

    def _current_ocr_mode(self) -> str:
        try:
            mode = str(get_config().get("mode", "local") or "local").lower()
        except Exception:
            mode = "local"
        return mode

    def _pending_hanwang_pages(self) -> list[Page]:
        return pending_ocr_pages(self._project)

    def _actionable_hanwang_pages(self) -> list[Page]:
        if self._project is None:
            return []
        return [
            page for page in self._project.pages
            if page_gate_info(page).action_enabled
        ]

    def _first_pending_hanwang_page(self) -> Page | None:
        pending = self._pending_hanwang_pages()
        return pending[0] if pending else None

    def _actionable_hanwang_pages_from(self, first_page: Page) -> list[Page]:
        """Return directly runnable Hanwang OCR pages with the submitted page first."""
        actionable = self._actionable_hanwang_pages()
        if not actionable and page_gate_info(first_page).action_enabled:
            return [first_page]
        first_number = first_page.page_number
        ordered = [page for page in actionable if page.page_number == first_number]
        ordered.extend(page for page in actionable if page.page_number != first_number)
        return ordered

    def _emit_page_gate_state(self, page: Page) -> None:
        gate = page_gate_info(page)
        reason_code = gate.reason_code
        reason_text = gate.reason_text
        if not self._pending_hanwang_pages() and gate.page_state == "ocr_complete":
            reason_code = "all_pages_done"
            reason_text = "全部已完成 OCR"
        self.page_gate_state.emit(page.page_number, gate.page_state, gate.is_pending, reason_code, reason_text)
        self.primary_action.emit(page.page_number, gate.action_key, gate.action_label, gate.action_enabled)

    def refresh_page_gate_states(self) -> None:
        if not self._project:
            return
        for page in self._project.pages:
            self._emit_page_gate_state(page)

    def handle_block_contract_changed(self, page_number: int, change_kind: str) -> None:
        page = self.page_by_number(page_number)
        if page is None:
            return
        had_ocr = page_has_ocr_result(page) or page_is_ocr_done(page)
        for block in page_layout_blocks(page):
            mark_ocr_text_invalidated(block, change_kind)
        if had_ocr:
            invalidate_page_ocr(page, change_kind)
        mark_page_layout_done(page)
        self.set_current_page_number(page_number)
        self._update_max_step()
        self._emit_page_gate_state(page)
        if self._store:
            self.save_project()

    def handle_ocr_entry_requested(self, source: str, page_number: int) -> None:
        if not self.is_hanwang_mode():
            return
        if not self._project or not self._project.pages:
            self.status_message.emit("当前没有可 OCR 的页面")
            return
        pending_page = self._first_pending_hanwang_page()
        if source != "layout_submit":
            current = self.page_by_number(page_number)
            actionable = self._actionable_hanwang_pages()
            if actionable:
                target = current if current in actionable else actionable[0]
                targets = self._actionable_hanwang_pages_from(target)
                self.start_ocr(targets, target_page_numbers={page.page_number for page in targets})
                return
            if pending_page is None:
                current = current or self._project.pages[0]
                self._emit_page_gate_state(current)
                self.status_message.emit("全部已完成 OCR")
                return
            self.set_current_page_number(pending_page.page_number)
            self.focus_page.emit(pending_page.page_number)
            self.step_requested.emit(STEP_LAYOUT)
            self._emit_page_gate_state(pending_page)
            self.status_message.emit(page_gate_info(pending_page).reason_text)
            return

        target = self.page_by_number(page_number) or pending_page
        if target is None:
            self.status_message.emit("全部已完成 OCR")
            return
        gate = page_gate_info(target)
        if not gate.action_enabled:
            actionable = self._actionable_hanwang_pages()
            if gate.reason_code == "ocr_complete" and actionable:
                next_target = actionable[0]
                targets = self._actionable_hanwang_pages_from(next_target)
                self.start_ocr(targets, target_page_numbers={page.page_number for page in targets})
                return
            self._emit_page_gate_state(target)
            self.status_message.emit("全部已完成 OCR" if gate.reason_code == "ocr_complete" else gate.reason_text)
            return
        if gate.page_state == "layout_pending":
            self.focus_page.emit(target.page_number)
            self.step_requested.emit(STEP_LAYOUT)
            self._emit_page_gate_state(target)
            self.status_message.emit(gate.reason_text)
            return
        targets = self._actionable_hanwang_pages_from(target)
        if not targets:
            self.status_message.emit("当前没有可 OCR 的页面")
            return
        self.start_ocr(targets, target_page_numbers={page.page_number for page in targets})

    def _layout_status_label(self) -> str:
        return "版面分析"

    def _ocr_status_label(self) -> str:
        return "文字识别"

    def _proof_ocr_status_label(self, engine: object | None = None) -> str:
        return self._ocr_status_label()

    def _ocr_page_concurrency(self) -> int:
        try:
            value = int(get_config().get("ocr_page_concurrency", 2))
        except (TypeError, ValueError):
            value = 2
        return max(1, min(20, value))

    def _effective_ocr_page_concurrency(self, pages: List[Page], engine: object) -> int:
        if not bool(getattr(engine, "prefer_page_hybrid_blocks", False)):
            return 1
        return max(1, min(len(pages), self._ocr_page_concurrency()))

    @staticmethod
    def _ocr_public_progress_message(progress: OcrProgress) -> str:
        raw = progress.message or ""
        total = max(0, int(progress.total_pages))
        current = max(0, int(progress.current_page))
        if "失败" in raw:
            if total and current:
                return f"文字识别失败：第 {current}/{total} 页"
            return "文字识别失败"
        if "警告" in raw or "fallback" in raw.lower():
            return "文字识别完成，部分字框需要检查"
        return "文字识别中…"

    # ------------------------------------------------------------------ workflow actions

    def on_images_ready(self, pages: List[Page]) -> None:
        """导入图片/PDF 后的处理。"""
        if not self._project:
            self._project = OcrProject(name="未命名项目")

        self._project.pages = pages

        # 更新页面状态
        for page in pages:
            mark_page_imported(page)

        self._update_max_step()
        self.project_changed.emit(self._project)
        self.status_message.emit(f"已导入 {len(pages)} 页")

        if self._store:
            self.save_project()

    def on_layout_done(self, pages: List[Page]) -> None:
        """版面分析完成后的处理。"""
        self._project.pages = pages

        for page in pages:
            if page_has_error(page):
                mark_page_layout_failed(page)
            else:
                mark_page_layout_done(page)

        self._update_max_step()
        self.layout_finished.emit(pages)
        failed_pages = [page for page in pages if page_has_error(page)]
        success_count = len(pages) - len(failed_pages)
        if self._auto_start_ocr_after_layout:
            self.status_message.emit(
                f"版面分析完成：{success_count}/{len(pages)} 页成功"
                + (f"，{len(failed_pages)} 页失败" if failed_pages else "")
                + "，正在启动文字识别…"
            )
        else:
            self.status_message.emit(
                f"版面分析完成：{success_count}/{len(pages)} 页成功"
                + (f"，{len(failed_pages)} 页失败" if failed_pages else "")
            )

        if self._store:
            self.save_project()

        if self.is_hanwang_mode():
            self._auto_start_ocr_after_layout = False
            self._pending_layout_pages = None
            self._pending_proof_pages = None
            self.refresh_page_gate_states()
            return

        if self._discard_parallel_proof_result:
            self._pending_layout_pages = None
            self._pending_proof_pages = None
            self._discard_parallel_proof_result = False
            queued_callback = self._queued_ocr_progress_callback
            self._queued_ocr_progress_callback = None
            if self.get_text_ocr_block_count() == 0:
                self.status_message.emit(
                    "版面分析完成，但文字识别未获得可用文本"
                )
                return
            self.status_message.emit(
                "文字识别继续处理中…"
            )
            self.start_ocr(pages, notify_page_callback=queued_callback)
            return

        if self._worker_is_running(self._proof_ocr_worker) or self._pending_proof_pages is not None:
            self._pending_layout_pages = pages
            if self._pending_proof_pages is not None:
                self._finish_parallel_proof_ocr()
            else:
                self.status_message.emit(
                    f"版面分析完成：{len(pages)} 页，等待文字识别…"
                )
            return

        if self._auto_start_ocr_after_layout:
            queued_callback = self._queued_ocr_progress_callback
            self._auto_start_ocr_after_layout = False
            self._queued_ocr_progress_callback = None
            if self.get_text_ocr_block_count() == 0:
                self.status_message.emit("版面分析未产生可识别文字块，已停止自动 OCR")
                return
            self.start_ocr(pages, notify_page_callback=queued_callback)

    def on_ocr_done(self, pages: List[Page]) -> None:
        """OCR 识别完成后的处理。

        这是业务完成事件，由 Worker 触发。
        与用户点击"进入校对"按钮的导航意图严格分离。
        """
        target_page_numbers = self._ocr_target_page_numbers
        self._ocr_target_page_numbers = None
        if target_page_numbers is not None and self._project is not None:
            updated_by_number = {page.page_number: page for page in pages}
            pages = [
                updated_by_number.get(page.page_number, page)
                for page in self._project.pages
            ]
        self._project.pages = pages

        processed_pages = (
            pages
            if target_page_numbers is None
            else [page for page in pages if page.page_number in target_page_numbers]
        )
        for page in processed_pages:
            if page_has_error(page):
                mark_page_ocr_failed(page)
            else:
                mark_page_ocr_done(page)

        proof_stats = self._proof_crop_service.normalize_pages(processed_pages)

        # 自动标记低置信行
        flagged = self._proof_auto_flag_service.auto_flag(processed_pages)

        self._update_max_step()
        self.ocr_finished.emit(pages)
        failed_pages = [
            page for page in pages
            if page_has_error(page)
        ]
        fallback_warning = self._proof_fallback_warning(proof_stats, processed_pages)
        self.status_message.emit(
            f"文字识别完成：自动标记 {flagged} 行；可进入校对"
            + (f"（{len(failed_pages)} 页失败）" if failed_pages else "")
            + (f"；{fallback_warning}" if fallback_warning else "")
        )

        if self._store:
            self.save_project()
        if self.is_hanwang_mode():
            self.refresh_page_gate_states()

    # ------------------------------------------------------------------ worker management

    def start_layout_analysis(self, pages: List[Page]) -> bool:
        """启动版面分析 worker（带进度反馈）。"""
        if not pages:
            self.status_message.emit("当前没有可分析的页面")
            return False
        if self._layout_worker and self._layout_worker.isRunning():
            self.status_message.emit("版面分析仍在进行中…")
            return False
        if self._ocr_worker and self._ocr_worker.isRunning():
            self.status_message.emit("文字识别仍在进行中…")
            return False
        if self._proof_ocr_worker and self._proof_ocr_worker.isRunning():
            self.status_message.emit("文字识别仍在进行中…")
            return False
        from app.core.layout_analyzer import LayoutWorker
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        self._discard_parallel_proof_result = False
        parallel_started = False if self.is_hanwang_mode() else self._start_parallel_proof_ocr(pages)
        self._auto_start_ocr_after_layout = False if self.is_hanwang_mode() else not parallel_started
        self._queued_ocr_progress_callback = None
        self._layout_worker = LayoutWorker(pages)
        self._connect_worker_cleanup("_layout_worker", self._layout_worker)
        self._layout_worker.page_done.connect(self._on_layout_progress)
        stage_signal = getattr(self._layout_worker, "stage_update", None)
        if stage_signal is not None and hasattr(stage_signal, "connect"):
            stage_signal.connect(self._on_layout_stage)
        self._layout_worker.all_done.connect(self.on_layout_done)
        self._layout_worker.error.connect(self._on_worker_error)
        cancelled_signal = getattr(self._layout_worker, "cancelled", None)
        if cancelled_signal is not None and hasattr(cancelled_signal, "connect"):
            cancelled_signal.connect(self._on_layout_cancelled)
        self._layout_worker.start()
        if parallel_started:
            self.status_message.emit(
                "版面分析与文字识别同步执行中…"
            )
        else:
            self.status_message.emit(f"{self._layout_status_label()}中…")
        return True

    def _on_layout_progress(self, current: int, total: int) -> None:
        """版面分析进度更新。"""
        self.progress_state_changed.emit(WorkflowProgressState(
            phase="layout",
            current=current,
            total=total,
            message=f"{self._layout_status_label()}中…",
        ))
        self.layout_progress.emit(current, total)
        self.status_message.emit(f"{self._layout_status_label()}中…")

    def _on_layout_stage(self, current: int, total: int, message: str) -> None:
        self.progress_state_changed.emit(WorkflowProgressState(
            phase="layout",
            current=current,
            total=total,
            message=message,
        ))
        self.layout_stage.emit(current, total, message)

    def start_ocr(
        self,
        pages: List[Page],
        notify_page_callback: Callable = None,
        target_page_numbers: set[int] | None = None,
    ) -> bool:
        """启动 OCR worker（使用 OcrPipeline + engine adapter）。"""
        if self._ocr_worker and self._ocr_worker.isRunning():
            self.status_message.emit("文字识别仍在进行中…")
            return False
        if self._proof_ocr_worker and self._proof_ocr_worker.isRunning():
            self.status_message.emit("文字识别仍在进行中…")
            return False

        text_ocr_block_count = count_text_ocr_blocks(pages)
        if text_ocr_block_count == 0:
            self.worker_error.emit("当前没有可识别的文字块，请先完成版面分析或补充文字区域。")
            self.status_message.emit("没有可识别的文字块")
            return False

        # 根据配置创建引擎
        engine = create_engine()
        page_concurrency = self._effective_ocr_page_concurrency(pages, engine)
        pipeline = OcrPipeline(engine=engine, page_concurrency=page_concurrency)

        self._ocr_target_page_numbers = target_page_numbers
        self._last_ocr_progress_completed_pages = 0
        self._ocr_worker = OcrPipelineWorker(pipeline, pages)
        self._connect_worker_cleanup("_ocr_worker", self._ocr_worker)
        self._ocr_worker.progress_state.connect(self._on_ocr_progress)
        if notify_page_callback:
            self._ocr_worker.progress_update.connect(notify_page_callback)
        self._ocr_worker.all_done.connect(self.on_ocr_done)
        self._ocr_worker.error.connect(self._on_worker_error)
        self._on_ocr_progress(OcrProgress(
            current_page=1 if pages else 0,
            total_pages=len(pages),
            current_block=0,
            total_blocks=0,
            completed_pages=0,
            message=f"文字识别准备中… 共 {len(pages)} 页",
        ))
        self._ocr_worker.start()
        self.step_requested.emit(STEP_OCR)
        self.status_message.emit("文字识别中…")
        return True

    def _start_parallel_proof_ocr(self, pages: List[Page]) -> bool:
        if self.is_hanwang_mode():
            return False
        engine = create_engine()
        if not bool(getattr(engine, "prefer_page_ocr", False)):
            return False
        proof_pages = self._clone_pages_for_parallel_proof(pages)
        pipeline = OcrPipeline(engine=engine)
        self._last_ocr_progress_completed_pages = 0
        self._proof_ocr_worker = OcrPipelineWorker(pipeline, proof_pages)
        self._connect_worker_cleanup("_proof_ocr_worker", self._proof_ocr_worker)
        self._proof_ocr_worker.progress_state.connect(self._on_ocr_progress)
        self._proof_ocr_worker.all_done.connect(self._on_parallel_proof_done)
        self._proof_ocr_worker.error.connect(self._on_worker_error)
        self._proof_ocr_worker.start()
        return True

    def _clone_pages_for_parallel_proof(self, pages: List[Page]) -> List[Page]:
        proof_pages = copy.deepcopy(pages)
        for index, (source_page, proof_page) in enumerate(zip(pages, proof_pages)):
            page_key = self._make_parallel_page_key(source_page, index)
            setattr(source_page, PARALLEL_PROOF_PAGE_KEY_ATTR, page_key)
            setattr(proof_page, PARALLEL_PROOF_PAGE_KEY_ATTR, page_key)
        for page in proof_pages:
            replace_page_layout_blocks(page, [
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox(0, 0, max(1, int(page.width)), max(1, int(page.height))),
                    order=0,
                    note=f"Parallel {self._proof_ocr_status_label()} container",
                )
            ])
        return proof_pages

    def _on_parallel_proof_done(self, pages: List[Page]) -> None:
        if self._discard_parallel_proof_result:
            self._pending_proof_pages = None
            return
        self._pending_proof_pages = pages
        if self._pending_layout_pages is not None:
            self._finish_parallel_proof_ocr()

    def _collect_proof_lines(self, page: Page):
        return [
            occurrence.line
            for occurrence in iter_page_ocr_line_occurrences(page)
        ]

    def _make_parallel_page_key(self, page: Page, index: int) -> tuple[int, str, str, int, int]:
        return (
            int(index),
            page.display_image_path,
            page.source_path,
            int(page.source_page_index),
            int(page.page_number),
        )

    def _parallel_page_key(self, page: Page) -> tuple[int, str, str, int, int]:
        key = getattr(page, PARALLEL_PROOF_PAGE_KEY_ATTR, None)
        if key is not None:
            return key
        return self._make_parallel_page_key(page, -1)

    def _finish_parallel_proof_ocr(self) -> None:
        if self._pending_layout_pages is None or self._pending_proof_pages is None:
            return
        layout_pages = self._pending_layout_pages
        proof_pages = self._pending_proof_pages
        assigner = OcrPipeline(engine=object())
        proof_by_key: dict[tuple[int, str, str, int, int], Page] = {}
        duplicate_keys: set[tuple[int, str, str, int, int]] = set()
        for page in proof_pages:
            key = self._parallel_page_key(page)
            if key in proof_by_key:
                duplicate_keys.add(key)
                logger.warning(
                    "Parallel proof OCR duplicate page identity; skipping ambiguous proof merge for page=%s",
                    page.display_image_path,
                )
                continue
            proof_by_key[key] = page
        for layout_page in layout_pages:
            layout_key = self._parallel_page_key(layout_page)
            if layout_key in duplicate_keys:
                logger.warning(
                    "Parallel proof OCR ambiguous page identity; skipping proof merge for page=%s",
                    layout_page.display_image_path,
                )
                continue
            proof_page = proof_by_key.get(layout_key)
            if proof_page is None:
                logger.warning(
                    "Parallel proof OCR missing page result; skipping proof merge for page=%s",
                    layout_page.display_image_path,
                )
                continue
            assigner.assign_page_ocr_lines_to_blocks(
                layout_page,
                self._collect_proof_lines(proof_page),
            )
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        self.on_ocr_done(layout_pages)

    def _on_ocr_progress(self, progress: OcrProgress) -> None:
        completed_pages = max(0, int(progress.completed_pages))
        if completed_pages != self._last_ocr_progress_completed_pages and self._project:
            for page in self._project.pages:
                reconcile_page_ocr_done_from_result(page)
            if project_has_any_ocr_done_page(self._project) and self._max_step < STEP_VPROOF:
                self._update_max_step()
            self._last_ocr_progress_completed_pages = completed_pages
        self.progress_state_changed.emit(WorkflowProgressState(
            phase="ocr",
            completed_pages=completed_pages,
            total_pages=int(progress.total_pages),
            message=self._ocr_public_progress_message(progress),
        ))
        self.ocr_progress.emit(progress)
        if progress.message:
            self.status_message.emit(self._ocr_public_progress_message(progress))

    @staticmethod
    def _proof_fallback_warning(stats, pages: list[Page] | None = None) -> str:
        fallback_total = (
            int(getattr(stats, "fallback_chars", 0))
            + int(getattr(stats, "unavailable_chars", 0))
        )
        fallback_lines = int(getattr(stats, "fallback_lines", 0))
        if fallback_total <= 0 and pages:
            seen_lines: set[int] = set()
            for page in pages:
                for _block, line, _line_idx in iter_unique_page_text_lines(page):
                    line_fallback_chars = 0
                    for char in line_ocr_chars(line):
                        source = (char.bbox_source or "").strip().lower()
                        granularity = (char.bbox_granularity or "").strip().lower()
                        if source in {"fallback", "unavailable"} or granularity in {"fallback", "unavailable", "line"}:
                            line_fallback_chars += 1
                    if line_fallback_chars:
                        fallback_total += line_fallback_chars
                        if id(line) not in seen_lines:
                            fallback_lines += 1
                            seen_lines.add(id(line))
        if fallback_total <= 0:
            return ""
        return (
            f"警告：proof fallback {fallback_lines} 行/"
            f"{fallback_total} 字，字框为估算或不可用"
        )

    def _on_worker_error(self, msg: str) -> None:
        self._discard_parallel_proof_result = True
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        self._ocr_target_page_numbers = None
        logger.error("Worker error: %s", msg)
        self.worker_error.emit(msg)
        self.status_message.emit("处理失败")

    def _on_layout_cancelled(self) -> None:
        self._discard_parallel_proof_result = True
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        self.set_layout_run_enabled(True)
        self.layout_cancelled.emit()
        self.status_message.emit("版面分析已取消")

    def _connect_worker_cleanup(self, attr_name: str, worker) -> None:
        finished = getattr(worker, "finished", None)
        if finished is None or not hasattr(finished, "connect"):
            return

        def cleanup_finished(_attr=attr_name, _worker=worker) -> None:
            self._clear_worker_ref(_attr, _worker)

        finished.connect(cleanup_finished)

    def _request_worker_stop(self, worker, *, wait_ms: int, force: bool) -> bool:
        if worker is None:
            return True
        try:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
            elif hasattr(worker, "requestInterruption"):
                worker.requestInterruption()
        except RuntimeError:
            return True
        if wait_ms > 0:
            try:
                if worker.wait(wait_ms):
                    return True
            except RuntimeError:
                return True
        if force:
            try:
                if worker.isRunning():
                    worker.terminate()
                    worker.wait(1000)
            except RuntimeError:
                return True
        return not self._worker_is_running(worker)

    @staticmethod
    def _progress_stage_key(message: str) -> str:
        lowered = (message or "").lower()
        if "pp-ocrv5" in lowered and "complete" in lowered:
            return "ppocr_done"
        if "pp-ocrv5" in lowered or "prepass" in lowered:
            return "ppocr"
        if "segimg" in lowered or "分块" in message:
            return "segimg"
        if "recog 准备" in lowered:
            return "recog_prepare"
        if "hanwang ocr" in lowered:
            return "hanwang_recog"
        if "已写回" in message:
            return "writeback"
        return lowered[:48]

    def _clear_worker_ref(self, attr_name: str, worker) -> None:
        if getattr(self, attr_name, None) is worker:
            setattr(self, attr_name, None)

    def _worker_is_running(self, worker) -> bool:
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False


class OcrPipelineWorker(QThread):
    """基于 OcrPipeline 的 OCR worker thread。"""

    page_done = Signal(int, int)     # (page_idx, total_pages)
    all_done  = Signal(list)         # List[Page]
    progress_update = Signal(int, int)
    progress_state = Signal(object)  # OcrProgress
    error     = Signal(str)

    def __init__(self, pipeline: OcrPipeline, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._pages = pages

    def run(self) -> None:
        try:
            total = len(self._pages)
            project = OcrProject(name="_ocr_worker_", pages=self._pages)
            completed_pages = 0
            progress_lock = Lock()
            last_progress_emit = 0.0
            last_progress_signature: tuple[int, int, int, int, str] | None = None

            def on_progress(progress: OcrProgress):
                nonlocal completed_pages, last_progress_emit, last_progress_signature
                with progress_lock:
                    now = time.monotonic()
                    message = progress.message or ""
                    page_advanced = progress.completed_pages > completed_pages
                    progress_bucket = (
                        int(progress.current_block / max(1, progress.total_blocks) * 100)
                        if progress.total_blocks > 0
                        else int(progress.current_block)
                    )
                    signature = (
                        int(progress.current_page),
                        int(progress.total_pages),
                        progress_bucket,
                        int(progress.total_blocks),
                        WorkflowController._progress_stage_key(message),
                    )
                    stage_changed = signature != last_progress_signature
                    important = (
                        page_advanced
                        or stage_changed
                        or progress.current_block <= 0
                        or "失败" in message
                        or "警告" in message
                    )
                    if important or now - last_progress_emit >= OCR_PROGRESS_MIN_EMIT_INTERVAL_SECONDS:
                        self.progress_state.emit(progress)
                        last_progress_emit = now
                        last_progress_signature = signature
                    while completed_pages < progress.completed_pages:
                        self.progress_update.emit(completed_pages, total)
                        self.page_done.emit(completed_pages, total)
                        completed_pages += 1

            result = self._pipeline.process_project(project, progress_callback=on_progress)

            self.all_done.emit(result.pages)
        except Exception as e:
            logger.error("OCR worker failed: %s", e)
            self.error.emit(str(e))
