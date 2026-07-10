"""Managed working-cache and user snapshot boundary for OCR projects."""
from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from app.core import quality_probe as qp
from app.core.app_config import AppConfig
from app.core.project_store import ProjectDataError, ProjectStore
from app.models import OcrProject


_ACTIVE_WORKSPACE_KEY = "active_workspace_db_path"
_WORKSPACE_DB_NAME = "working.ocrproj"


@dataclass(frozen=True)
class WorkingProject:
    """The active cache-backed project and its open store."""

    store: ProjectStore
    project: OcrProject


class ProjectWorkspaceService:
    """Own cache lifecycle and user-facing project snapshots.

    A working database is the active project's persistence target.  A user
    selected ``.ocrproj`` is never adopted as that target: it is written as a
    new single-project snapshot, then left untouched by later auto-saves.
    """

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        settings: AppConfig | None = None,
    ) -> None:
        self._root = Path(workspace_root) if workspace_root is not None else self.default_workspace_root()
        self._settings = settings or AppConfig.instance()

    @staticmethod
    def default_workspace_root() -> Path:
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata) / "ocr_process" / "workspaces"
        value = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
        base = Path(value) if value else Path.home() / ".ocr_process"
        return base / "workspaces"

    def create_working_project(self, name: str) -> WorkingProject:
        db_path = self._new_workspace_db_path()
        store = ProjectStore(str(db_path))
        store.open()
        project = store.save_project(OcrProject(name=name, db_path=str(db_path)))
        self._remember_active_workspace(db_path)
        return WorkingProject(store=store, project=project)

    def resume_working_project(self) -> WorkingProject | None:
        value = str(self._settings.get(_ACTIVE_WORKSPACE_KEY, "") or "").strip()
        if not value:
            return None
        db_path = Path(value)
        if not db_path.is_file():
            self._settings.set(_ACTIVE_WORKSPACE_KEY, "")
            return None
        try:
            return self._open_single_project_database(db_path, remember=True)
        except Exception:
            self._settings.set(_ACTIVE_WORKSPACE_KEY, "")
            raise

    def open_snapshot_into_workspace(self, snapshot_path: str | Path) -> WorkingProject:
        source_path = Path(snapshot_path).resolve()
        if not source_path.is_file():
            raise ProjectDataError(f"project snapshot does not exist: {source_path}")

        source = self._open_single_project_database(source_path, remember=False)
        source.store.close()

        workspace_dir = self._new_workspace_dir()
        workspace_path = workspace_dir / _WORKSPACE_DB_NAME
        shutil.copy2(source_path, workspace_path)
        self._copy_snapshot_sidecars_to_workspace(source_path, workspace_path)

        try:
            return self._open_single_project_database(workspace_path, remember=True)
        except Exception:
            shutil.rmtree(workspace_dir, ignore_errors=True)
            raise

    def write_snapshot(
        self,
        *,
        active_store: ProjectStore,
        project: OcrProject,
        target_path: str | Path,
    ) -> Path:
        """Write a complete new snapshot without changing the active cache.

        The active cache is flushed first.  The staged SQLite file contains one
        project and copies page/thumbnail/raw-artifact files into a sibling
        ``<name>.assets`` directory before it replaces the requested target.
        """
        target = Path(target_path).resolve()
        if target.suffix.lower() != ".ocrproj":
            target = target.with_suffix(".ocrproj")
        source = Path(active_store.db_path).resolve()
        if target == source:
            raise ProjectDataError("project snapshot target cannot be the active working cache")
        target.parent.mkdir(parents=True, exist_ok=True)

        active_store.save_project(project)
        active_store.flush_to_main_file()

        stage_dir = target.parent / f".{target.stem}.snapshot-stage-{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=True)
        stage_db = stage_dir / target.name
        stage_assets = stage_dir / self._asset_dir_name(target)
        try:
            self._copy_database(source, stage_db)
            self._materialize_snapshot_assets(
                source_db=source,
                stage_db=stage_db,
                stage_assets=stage_assets,
                asset_dir_name=self._asset_dir_name(target),
            )
            self._copy_quality_probe_sidecar(source, stage_db)
            self._validate_snapshot(stage_db)
            self._install_snapshot_stage(stage_dir, target)
            return target
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def forget_working_project(self, db_path: str | Path | None) -> None:
        current = str(self._settings.get(_ACTIVE_WORKSPACE_KEY, "") or "").strip()
        if db_path is None or not current:
            return
        try:
            if Path(current).resolve() == Path(db_path).resolve():
                self._settings.set(_ACTIVE_WORKSPACE_KEY, "")
        except OSError:
            self._settings.set(_ACTIVE_WORKSPACE_KEY, "")

    def _new_workspace_dir(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        directory = self._root / f"workspace-{uuid.uuid4().hex}"
        directory.mkdir()
        return directory

    def _new_workspace_db_path(self) -> Path:
        return self._new_workspace_dir() / _WORKSPACE_DB_NAME

    def _remember_active_workspace(self, db_path: Path) -> None:
        self._settings.set(_ACTIVE_WORKSPACE_KEY, str(db_path.resolve()))

    def _open_single_project_database(
        self,
        db_path: Path,
        *,
        remember: bool,
    ) -> WorkingProject:
        store = ProjectStore(str(db_path))
        try:
            store.open()
            projects = store.list_projects()
            if len(projects) != 1:
                raise ProjectDataError(
                    f"project snapshot must contain exactly one project, found {len(projects)}"
                )
            project = store.load_project(project_id=int(projects[0]["id"]))
            if project is None:
                raise ProjectDataError("project snapshot has no readable project")
            project.db_path = str(db_path)
            self._require_page_assets(project)
            if remember:
                self._remember_active_workspace(db_path)
            return WorkingProject(store=store, project=project)
        except Exception:
            store.close()
            raise

    @staticmethod
    def _copy_database(source: Path, target: Path) -> None:
        source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        target_conn = sqlite3.connect(target)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
            source_conn.close()

    def _materialize_snapshot_assets(
        self,
        *,
        source_db: Path,
        stage_db: Path,
        stage_assets: Path,
        asset_dir_name: str,
    ) -> None:
        stage_assets.mkdir(parents=True)
        conn = sqlite3.connect(stage_db)
        conn.row_factory = sqlite3.Row
        try:
            pages = conn.execute(
                "SELECT id, uid, image_path, cache_image_path, thumbnail_path FROM page ORDER BY id"
            ).fetchall()
            for row in pages:
                page_uid = str(row["uid"] or row["id"])
                image_source = self._resolve_source_path(
                    source_db,
                    str(row["cache_image_path"] or row["image_path"] or ""),
                )
                if not image_source.is_file():
                    raise ProjectDataError(f"working page image is missing: {image_source}")
                image_relative = self._copy_asset(
                    image_source,
                    stage_assets,
                    asset_dir_name,
                    "images",
                    page_uid,
                )
                thumb_value = str(row["thumbnail_path"] or "")
                thumbnail_relative = ""
                if thumb_value:
                    thumbnail_source = self._resolve_source_path(source_db, thumb_value)
                    if not thumbnail_source.is_file():
                        raise ProjectDataError(f"working thumbnail is missing: {thumbnail_source}")
                    thumbnail_relative = self._copy_asset(
                        thumbnail_source,
                        stage_assets,
                        asset_dir_name,
                        "thumbnails",
                        page_uid,
                    )
                conn.execute(
                    "UPDATE page SET image_path=?, cache_image_path=?, thumbnail_path=? WHERE id=?",
                    (image_relative, image_relative, thumbnail_relative, row["id"]),
                )

            artifacts = conn.execute(
                "SELECT id, uid, artifact_path FROM raw_ocr_artifact WHERE artifact_path <> ''"
            ).fetchall()
            for row in artifacts:
                artifact_source = self._resolve_source_path(source_db, str(row["artifact_path"] or ""))
                if not artifact_source.is_file():
                    raise ProjectDataError(f"working OCR artifact is missing: {artifact_source}")
                artifact_relative = self._copy_asset(
                    artifact_source,
                    stage_assets,
                    asset_dir_name,
                    "artifacts",
                    str(row["uid"] or row["id"]),
                )
                conn.execute(
                    "UPDATE raw_ocr_artifact SET artifact_path=? WHERE id=?",
                    (artifact_relative, row["id"]),
                )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _resolve_source_path(source_db: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else source_db.parent / path

    @staticmethod
    def _copy_asset(
        source: Path,
        stage_assets: Path,
        asset_dir_name: str,
        category: str,
        stable_name: str,
    ) -> str:
        suffix = source.suffix or ".bin"
        relative = Path(asset_dir_name) / category / f"{stable_name}{suffix}"
        destination = stage_assets.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return relative.as_posix()

    @staticmethod
    def _copy_quality_probe_sidecar(source_db: Path, stage_db: Path) -> None:
        source_sidecar = Path(str(source_db) + ".qprobe.json")
        if source_sidecar.is_file():
            shutil.copy2(source_sidecar, Path(str(stage_db) + ".qprobe.json"))

    def _copy_snapshot_sidecars_to_workspace(self, snapshot_db: Path, workspace_db: Path) -> None:
        source_assets = snapshot_db.parent / self._asset_dir_name(snapshot_db)
        if source_assets.exists():
            shutil.copytree(source_assets, workspace_db.parent / source_assets.name)
        source_sidecar = Path(str(snapshot_db) + ".qprobe.json")
        if source_sidecar.is_file():
            shutil.copy2(source_sidecar, Path(str(workspace_db) + ".qprobe.json"))

    def _validate_snapshot(self, stage_db: Path) -> None:
        working = self._open_single_project_database(stage_db, remember=False)
        try:
            self._require_page_assets(working.project)
        finally:
            working.store.close()

    @staticmethod
    def _require_page_assets(project: OcrProject) -> None:
        for page in project.pages:
            image_path = Path(page.display_image_path)
            if not image_path.is_file():
                raise ProjectDataError(f"project page image is missing: {image_path}")

    def _install_snapshot_stage(self, stage_dir: Path, target: Path) -> None:
        stage_db = stage_dir / target.name
        stage_assets = stage_dir / self._asset_dir_name(target)
        stage_sidecar = Path(str(stage_db) + ".qprobe.json")
        target_assets = target.parent / self._asset_dir_name(target)
        target_sidecar = Path(str(target) + ".qprobe.json")
        token = uuid.uuid4().hex
        backups = {
            target: target.parent / f".{target.name}.backup-{token}",
            target_assets: target.parent / f".{target_assets.name}.backup-{token}",
            target_sidecar: target.parent / f".{target_sidecar.name}.backup-{token}",
        }
        moved_targets: list[Path] = []
        installed: list[Path] = []
        try:
            for original, backup in backups.items():
                if original.exists():
                    os.replace(original, backup)
                    moved_targets.append(original)
            os.replace(stage_assets, target_assets)
            installed.append(target_assets)
            if stage_sidecar.exists():
                os.replace(stage_sidecar, target_sidecar)
                installed.append(target_sidecar)
            os.replace(stage_db, target)
            installed.append(target)
        except Exception:
            for installed_path in reversed(installed):
                self._remove_path(installed_path)
            for original in reversed(moved_targets):
                backup = backups[original]
                if backup.exists():
                    os.replace(backup, original)
            raise
        else:
            for backup in backups.values():
                self._remove_path(backup)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    @staticmethod
    def _asset_dir_name(snapshot_path: Path) -> str:
        return f"{snapshot_path.stem}.assets"


__all__ = ["ProjectWorkspaceService", "WorkingProject"]
