"""Application use cases and immutable UI contracts."""

from .contracts import (
    BlockView,
    LayoutEditCommand,
    LayoutEditResult,
    LayoutWorkspaceView,
    PageView,
    ProofEditCommand,
    ProofEditResult,
)
from .layout_workspace import build_layout_workspace_view
from .proof_workspace import ProofWorkspaceView, build_proof_workspace_view
from .workbench import WorkbenchApplication

__all__ = [
    "BlockView",
    "LayoutEditCommand",
    "LayoutEditResult",
    "LayoutWorkspaceView",
    "PageView",
    "ProofEditCommand",
    "ProofEditResult",
    "ProofWorkspaceView",
    "WorkbenchApplication",
    "build_layout_workspace_view",
    "build_proof_workspace_view",
]
