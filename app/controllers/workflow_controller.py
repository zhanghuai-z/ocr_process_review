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
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.core.logging import get_logger
from app.core.project_store import ProjectStore
from app.core.proof_engine import ProofEngine
from app.core import quality_probe as qp
from app.engines.real_ocr_adapter import create_engine
from app.models import (
    BBox, Block, BlockType, OcrProject, Page, PageStatus, ProofStatus,
)
from app.services.ocr_pipeline import OcrPipeline, OcrProgress
from app.services.proof_crop_service import ProofCropService

logger = get_logger(__name__)
PARALLEL_PROOF_PAGE_KEY_ATTR = "_parallel_proof_page_key"

# 步骤索引（与 stacked widget 顺序一致）
STEP_IMPORT = 0
STEP_LAYOUT = 1
STEP_OCR = 2
STEP_HPROOF = 3
STEP_VPROOF = 4


class WorkflowController(QObject):
    """工作流控制器。"""

    # 状态变化信号
    project_changed = Signal(object)       # OcrProject
    step_enabled_changed = Signal(int)     # 允许的最大步骤
    step_requested = Signal(int)           # 请求跳转步骤
    layout_finished = Signal(object)       # List[Page]
    ocr_finished = Signal(object)          # List[Page]
    ocr_progress = Signal(object)          # OcrProgress
    worker_error = Signal(str)             # 错误消息
    status_message = Signal(str)           # 状态栏消息

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Optional[OcrProject] = None
        self._store: Optional[ProjectStore] = None
        self._proof_engine = ProofEngine()
        self._proof_crop_service = ProofCropService()
        self._max_step: int = STEP_IMPORT
        self._layout_worker = None
        self._ocr_worker = None
        self._proof_ocr_worker = None
        self._pending_layout_pages: Optional[List[Page]] = None
        self._pending_proof_pages: Optional[List[Page]] = None
        self._auto_start_ocr_after_layout = True
        self._queued_ocr_progress_callback: Optional[Callable] = None

    # ------------------------------------------------------------------ project management

    @property
    def project(self) -> Optional[OcrProject]:
        return self._project

    @property
    def store(self) -> Optional[ProjectStore]:
        return self._store

    @property
    def proof_engine(self) -> ProofEngine:
        return self._proof_engine

    @property
    def max_step(self) -> int:
        return self._max_step

    def ensure_transient_project(self, name: str = "未命名项目") -> OcrProject:
        """确保存在一个临时项目对象，便于未保存状态下也能走通工作流。"""
        if self._project is None:
            self._project = OcrProject(name=name)
            self.project_changed.emit(self._project)
        return self._project

    def get_open_step(self) -> int:
        """打开项目后的推荐落点。"""
        if self._max_step >= STEP_HPROOF:
            return STEP_HPROOF
        if self._max_step >= STEP_LAYOUT:
            return STEP_LAYOUT
        return STEP_IMPORT

    def get_recognizable_block_count(self) -> int:
        if not self._project:
            return 0
        return sum(len(page.recognizable_blocks) for page in self._project.pages)

    def new_project(self, name: str, db_path: str) -> bool:
        """创建新项目。"""
        try:
            if self._store:
                self._store.close()
            self._store = ProjectStore(db_path)
            self._store.open()
            self._project = OcrProject(name=name, db_path=db_path)
            self._project = self._store.save_project(self._project)
            self._max_step = STEP_IMPORT
            # 切换项目：必须清空全局评测状态，避免旧项目的 probe 串到新项目
            qp.reset_active_store()
            self.project_changed.emit(self._project)
            self.step_enabled_changed.emit(self._max_step)
            self.status_message.emit(f"新建项目：{db_path}")
            return True
        except Exception as e:
            logger.error("Failed to create project: %s", e)
            self.worker_error.emit(f"创建项目失败：{e}")
            return False

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
            self._proof_crop_service.normalize_project(self._project)

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
            self.status_message.emit(f"已打开：{db_path}")
            return True
        except Exception as e:
            logger.error("Failed to open project: %s", e)
            self.worker_error.emit(f"打开项目失败：{e}")
            return False

    def save_project(self) -> bool:
        """保存项目。"""
        if not self._project or not self._store:
            return False
        try:
            self._store.save_project(self._project)
            # 副带保存评测 sidecar（如果当前有 active store）
            store = qp.get_active_store()
            if store is not None and self._project.db_path:
                side = qp.sidecar_path_for_project(self._project.db_path)
                if side:
                    qp.save_store_to_path(store, side)
            self.status_message.emit("项目已保存")
            return True
        except Exception as e:
            logger.error("Save failed: %s", e)
            self.worker_error.emit(f"保存失败：{e}")
            return False

    def auto_save(self) -> None:
        """校对时自动保存修改的行。"""
        if not self._project or not self._store:
            return
        try:
            for page in self._project.pages:
                for block in page.blocks:
                    for line in block.lines:
                        if line.id and line.proof_status in (
                            ProofStatus.MODIFIED, ProofStatus.OK
                        ):
                            self._store.update_line(line)
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
        if not self._project.ocr_completed:
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
        # 进程退出：清空全局评测状态
        qp.reset_active_store()

    # ------------------------------------------------------------------ step management

    def can_enter_step(self, step: int) -> bool:
        """检查是否允许进入某步骤。"""
        return step <= self._max_step

    def request_step(self, step: int) -> None:
        """请求跳转到某步骤（由 UI 触发）。"""
        if not self.can_enter_step(step):
            self.status_message.emit("当前状态不允许进入该步骤")
            return
        self.step_requested.emit(step)

    def _compute_max_step(self) -> int:
        """根据项目状态计算可进入的最大步骤。"""
        if not self._project or not self._project.pages:
            return STEP_IMPORT

        pages = self._project.pages
        has_blocks = any(p.is_analyzed for p in pages)
        has_ocr = any(p.total_lines > 0 for p in pages)

        if has_ocr:
            return STEP_VPROOF
        if has_blocks:
            return STEP_OCR
        return STEP_LAYOUT

    def _update_max_step(self) -> None:
        """更新最大可进入步骤并通知 UI。"""
        self._max_step = self._compute_max_step()
        self.step_enabled_changed.emit(self._max_step)

    # ------------------------------------------------------------------ workflow actions

    def on_images_ready(self, pages: List[Page]) -> None:
        """导入图片/PDF 后的处理。"""
        if not self._project:
            self._project = OcrProject(name="未命名项目")

        self._project.pages = pages

        # 更新页面状态
        for page in pages:
            page.status = PageStatus.IMPORTED

        self._update_max_step()
        self.project_changed.emit(self._project)
        self.status_message.emit(f"已导入 {len(pages)} 页")

        if self._store:
            self.save_project()

    def on_layout_done(self, pages: List[Page]) -> None:
        """版面分析完成后的处理。"""
        self._layout_worker = None
        self._project.pages = pages

        for page in pages:
            page.status = PageStatus.ERROR if page.error_message else PageStatus.LAYOUT_DONE

        self._update_max_step()
        self.layout_finished.emit(pages)
        failed_pages = [page for page in pages if page.error_message]
        success_count = len(pages) - len(failed_pages)
        if self._auto_start_ocr_after_layout:
            self.status_message.emit(
                f"版面分析完成：{success_count}/{len(pages)} 页成功"
                + (f"，{len(failed_pages)} 页失败" if failed_pages else "")
                + "，正在启动 OCR…"
            )
        else:
            self.status_message.emit(
                f"版面分析完成：{success_count}/{len(pages)} 页成功"
                + (f"，{len(failed_pages)} 页失败" if failed_pages else "")
            )

        if self._store:
            self.save_project()

        if self._proof_ocr_worker or self._pending_proof_pages is not None:
            self._pending_layout_pages = pages
            if self._pending_proof_pages is not None:
                self._finish_parallel_proof_ocr()
            else:
                self.status_message.emit(f"版面分析完成：{len(pages)} 页，等待 PP-OCRv5 proof OCR…")
            return

        if self._auto_start_ocr_after_layout:
            queued_callback = self._queued_ocr_progress_callback
            self._auto_start_ocr_after_layout = False
            self._queued_ocr_progress_callback = None
            if self.get_recognizable_block_count() == 0:
                self.status_message.emit("版面分析未产生可识别文字块，已停止自动 OCR")
                return
            self.start_ocr(pages, notify_page_callback=queued_callback)

    def on_ocr_done(self, pages: List[Page]) -> None:
        """OCR 识别完成后的处理。

        这是业务完成事件，由 Worker 触发。
        与用户点击"进入校对"按钮的导航意图严格分离。
        """
        self._ocr_worker = None
        self._project.pages = pages

        for page in pages:
            page.status = PageStatus.OCR_DONE

        self._proof_crop_service.normalize_pages(pages)

        # 自动标记低置信行
        flagged = self._proof_engine.auto_flag(pages)

        self._update_max_step()
        self.ocr_finished.emit(pages)
        self.status_message.emit(f"识别完成，自动标记 {flagged} 行低置信度内容")
        self.step_requested.emit(STEP_HPROOF)

        if self._store:
            self.save_project()

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
            self.status_message.emit("OCR 识别仍在进行中…")
            return False
        if self._proof_ocr_worker and self._proof_ocr_worker.isRunning():
            self.status_message.emit("OCR 识别仍在进行中…")
            return False
        from app.core.layout_analyzer import LayoutWorker
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        parallel_started = self._start_parallel_proof_ocr(pages)
        self._auto_start_ocr_after_layout = not parallel_started
        self._queued_ocr_progress_callback = None
        self._layout_worker = LayoutWorker(pages)
        self._layout_worker.page_done.connect(self._on_layout_progress)
        self._layout_worker.all_done.connect(self.on_layout_done)
        self._layout_worker.error.connect(self._on_worker_error)
        self._layout_worker.start()
        if parallel_started:
            self.status_message.emit("版面分析与 PP-OCRv5 proof OCR 同步执行中…")
        else:
            self.status_message.emit("版面分析中…")
        return True

    def _on_layout_progress(self, current: int, total: int) -> None:
        """版面分析进度更新。"""
        self.status_message.emit(f"版面分析中… 第 {current + 1}/{total} 页")

    def start_ocr(self, pages: List[Page], notify_page_callback: Callable = None) -> bool:
        """启动 OCR worker（使用 OcrPipeline + engine adapter）。"""
        if self._ocr_worker and self._ocr_worker.isRunning():
            self.status_message.emit("OCR 识别仍在进行中…")
            return False
        if self._proof_ocr_worker and self._proof_ocr_worker.isRunning():
            self.status_message.emit("PP-OCRv5 proof OCR 仍在进行中…")
            return False

        recognizable_blocks = sum(len(page.recognizable_blocks) for page in pages)
        if recognizable_blocks == 0:
            self.worker_error.emit("当前没有可识别的文字块，请先完成版面分析或补充文字区域。")
            self.status_message.emit("没有可识别的文字块")
            return False

        # 根据配置创建引擎
        engine = create_engine()
        pipeline = OcrPipeline(engine=engine)

        self._ocr_worker = OcrPipelineWorker(pipeline, pages)
        self._ocr_worker.progress_state.connect(self._on_ocr_progress)
        if notify_page_callback:
            self._ocr_worker.progress_update.connect(notify_page_callback)
        self._ocr_worker.all_done.connect(self.on_ocr_done)
        self._ocr_worker.error.connect(self._on_worker_error)
        self._ocr_worker.start()
        self.step_requested.emit(STEP_OCR)
        self.status_message.emit("OCR 识别中…")
        return True

    def _start_parallel_proof_ocr(self, pages: List[Page]) -> bool:
        engine = create_engine()
        if not bool(getattr(engine, "prefer_page_ocr", False)):
            return False
        proof_pages = self._clone_pages_for_parallel_proof(pages)
        pipeline = OcrPipeline(engine=engine)
        self._proof_ocr_worker = OcrPipelineWorker(pipeline, proof_pages)
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
            page.blocks = [
                Block(
                    block_type=BlockType.TEXT,
                    bbox=BBox(0, 0, max(1, int(page.width)), max(1, int(page.height))),
                    order=0,
                    note="Parallel PP-OCRv5 proof OCR container",
                )
            ]
        return proof_pages

    def _on_parallel_proof_done(self, pages: List[Page]) -> None:
        self._proof_ocr_worker = None
        self._pending_proof_pages = pages
        if self._pending_layout_pages is not None:
            self._finish_parallel_proof_ocr()

    def _collect_proof_lines(self, page: Page):
        return [
            line
            for block in page.blocks
            for line in block.lines
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
        if self._project and any(page.total_lines > 0 for page in self._project.pages):
            if self._max_step < STEP_VPROOF:
                self._update_max_step()
        self.ocr_progress.emit(progress)
        if progress.message:
            self.status_message.emit(progress.message)

    def _on_worker_error(self, msg: str) -> None:
        self._layout_worker = None
        self._ocr_worker = None
        self._proof_ocr_worker = None
        self._pending_layout_pages = None
        self._pending_proof_pages = None
        logger.error("Worker error: %s", msg)
        self.worker_error.emit(msg)
        self.status_message.emit("处理失败")


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

            def on_progress(progress: OcrProgress):
                nonlocal completed_pages
                self.progress_state.emit(progress)
                while completed_pages < progress.completed_pages:
                    self.progress_update.emit(completed_pages, total)
                    self.page_done.emit(completed_pages, total)
                    completed_pages += 1

            result = self._pipeline.process_project(project, progress_callback=on_progress)

            self.all_done.emit(result.pages)
        except Exception as e:
            logger.error("OCR worker failed: %s", e)
            self.error.emit(str(e))
