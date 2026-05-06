"""工作流控制器：管理项目状态、步骤跳转、Worker 调度。

职责：
- 从 MainWindow 抽离项目创建/打开/保存逻辑
- 统一维护工作流状态机
- 根据状态启用/禁用步骤按钮
- 调度 LayoutWorker / OcrWorker
- 处理 OCR 完成后的状态流转
"""
from __future__ import annotations
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.core.logging import get_logger
from app.core.project_store import ProjectStore
from app.core.proof_engine import ProofEngine
from app.engines.real_ocr_adapter import create_engine
from app.models import (
    Block, BlockSource, OcrProject, Page, PageStatus, ProofStatus,
)
from app.services.ocr_pipeline import OcrPipeline, OcrProgress

logger = get_logger(__name__)

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
    worker_error = Signal(str)             # 错误消息
    status_message = Signal(str)           # 状态栏消息

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Optional[OcrProject] = None
        self._store: Optional[ProjectStore] = None
        self._proof_engine = ProofEngine()
        self._max_step: int = STEP_IMPORT

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

    def close(self) -> None:
        if self._store:
            self._store.close()

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

    def on_layout_done(self, pages: List[Page]) -> None:
        """版面分析完成后的处理。"""
        self._project.pages = pages

        for page in pages:
            page.status = PageStatus.LAYOUT_DONE

        self._update_max_step()
        self.layout_finished.emit(pages)
        self.status_message.emit(f"版面分析完成：{len(pages)} 页")

        if self._store:
            self.save_project()

    def on_ocr_done(self, pages: List[Page]) -> None:
        """OCR 识别完成后的处理。

        这是业务完成事件，由 Worker 触发。
        与用户点击"进入校对"按钮的导航意图严格分离。
        """
        self._project.pages = pages

        for page in pages:
            page.status = PageStatus.OCR_DONE

        # 自动标记低置信行
        flagged = self._proof_engine.auto_flag(pages)

        self._update_max_step()
        self.ocr_finished.emit(pages)
        self.status_message.emit(f"识别完成，自动标记 {flagged} 行低置信度内容")

        if self._store:
            self.save_project()

    # ------------------------------------------------------------------ worker management

    def start_layout_analysis(self, pages: List[Page]) -> None:
        """启动版面分析 worker。"""
        from app.core.layout_analyzer import LayoutWorker
        self._layout_worker = LayoutWorker(pages)
        self._layout_worker.all_done.connect(self.on_layout_done)
        self._layout_worker.error.connect(self._on_worker_error)
        self._layout_worker.start()
        self.status_message.emit("版面分析中…")

    def start_ocr(self, pages: List[Page], notify_page_callback: Callable = None) -> None:
        """启动 OCR worker（使用 OcrPipeline + engine adapter）。"""
        # 根据配置创建引擎
        engine = create_engine()
        pipeline = OcrPipeline(engine=engine)

        self._ocr_worker = OcrPipelineWorker(pipeline, pages)
        if notify_page_callback:
            self._ocr_worker.progress_update.connect(notify_page_callback)
        self._ocr_worker.all_done.connect(self.on_ocr_done)
        self._ocr_worker.error.connect(self._on_worker_error)
        self._ocr_worker.start()
        self.status_message.emit("OCR 识别中…")

    def _on_worker_error(self, msg: str) -> None:
        logger.error("Worker error: %s", msg)
        self.worker_error.emit(msg)
        self.status_message.emit("处理失败")


class OcrPipelineWorker(QThread):
    """基于 OcrPipeline 的 OCR worker thread。"""

    page_done = Signal(int, int)     # (page_idx, total_pages)
    all_done  = Signal(list)         # List[Page]
    progress_update = Signal(int, int)
    error     = Signal(str)

    def __init__(self, pipeline: OcrPipeline, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._pages = pages

    def run(self) -> None:
        try:
            total = len(self._pages)
            project = OcrProject(name="_ocr_worker_", pages=self._pages)

            def on_progress(progress: OcrProgress):
                self.progress_update.emit(progress.current_page - 1, total)
                self.page_done.emit(progress.current_page - 1, total)

            result = self._pipeline.process_project(project, progress_callback=on_progress)

            self.all_done.emit(result.pages)
        except Exception as e:
            logger.error("OCR worker failed: %s", e)
            self.error.emit(str(e))
