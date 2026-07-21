"""Application use cases and immutable UI contracts."""

from .contracts import (
    BlockView,
    ImportCompletionView,
    ImportFailureView,
    LayoutEditCommand,
    LayoutEditResult,
    LayoutWorkspaceView,
    PageView,
    ProofEditCommand,
    ProofEditResult,
)
from .layout_workspace import build_layout_workspace_view
from .ocr_workspace import (
    OcrAtomView,
    OcrLineView,
    OcrPageView,
    OcrRegionView,
    OcrWorkspaceView,
    build_ocr_workspace_view,
)
from .proof_workspace import ProofWorkspaceView, build_proof_workspace_view
from .workbench import WorkbenchApplication

__all__ = [
    "BlockView",
    "ImportCompletionView",
    "ImportFailureView",
    "LayoutEditCommand",
    "LayoutEditResult",
    "LayoutWorkspaceView",
    "PageView",
    "OcrAtomView",
    "OcrLineView",
    "OcrPageView",
    "OcrRegionView",
    "OcrWorkspaceView",
    "ProofEditCommand",
    "ProofEditResult",
    "ProofWorkspaceView",
    "WorkbenchApplication",
    "build_layout_workspace_view",
    "build_ocr_workspace_view",
    "build_proof_workspace_view",
]
