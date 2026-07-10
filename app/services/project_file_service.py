"""User-bound persistence for portable ``.ocrproj`` projects.

An active project is either an in-memory draft or is bound to exactly one
user-selected project file.  This service owns the bind/rebind operation and
the sibling asset directory.  It deliberately does not create a hidden project
database or a resumable working-project cache.
"""
from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core import quality_probe as qp
from app.core.project_store import ProjectDataError, ProjectStore
from app.models import OcrProject, Page
from app.models.layout_block_view import iter_page_layout_block_views
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.models.ocr_character_observation import line_ocr_chars_by_uid
from app.models.ocr_observation import block_ocr_line_observations_by_uid


@dataclass(frozen=True)
class BoundProject:
    """The open store and model bound to one user project file."""

    store: ProjectStore
    project: OcrProject


class ProjectFileService:
    """Open, bind, and materialize project-owned assets."""

    def open_project(self, file_path: str | Path) -> BoundProject:
        path = self._normalize_path(file_path)
        if not path.is_file():
            raise ProjectDataError(f"project file does not exist: {path}")
        store = ProjectStore(str(path))
        try:
            store.open()
            project = self._load_single_project(store)
            self._require_assets(project)
            project.db_path = str(path)
            return BoundProject(store=store, project=project)
        except Exception:
            store.close()
            raise

    def bind_project(self, project: OcrProject, file_path: str | Path) -> BoundProject:
        """Atomically create a user project file and bind ``project`` to it.

        The active object graph itself is written to a staged database.  This
        is intentional: OCR observations are currently held in uid-keyed
        runtime stores, so a deep-copied project alone would not include the
        same observations.  All mutated persistence fields are restored if
        staging fails; after success their ids belong to the newly bound file.
        """
        target = self._normalize_path(file_path)
        if project.db_path and Path(project.db_path).resolve() == target:
            raise ProjectDataError("binding target is already the active project file")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = target.parent / f".{target.stem}.save-stage-{uuid.uuid4().hex}"
        stage_dir.mkdir()
        stage_db = stage_dir / target.name
        asset_dir_name = self.asset_dir_name(target)
        stage_assets = stage_dir / asset_dir_name
        state = self._capture_mutable_state(project)
        stage_store: ProjectStore | None = None
        bound_store: ProjectStore | None = None
        try:
            self._materialize_assets_to_stage(project, stage_assets, asset_dir_name)
            self._clear_storage_ids(project)
            stage_store = ProjectStore(str(stage_db))
            stage_store.open()
            stage_store.save_project(project)
            self._write_quality_probe_sidecar(stage_db)
            stage_store.close()
            stage_store = None
            self._validate_stage(stage_db)
            self._install_stage(stage_dir, target)
            self._activate_bound_paths(project, target)
            project.db_path = str(target)
            bound_store = ProjectStore(str(target))
            bound_store.open()
            return BoundProject(store=bound_store, project=project)
        except Exception:
            if bound_store is not None:
                bound_store.close()
            self._restore_mutable_state(state)
            raise
        finally:
            if stage_store is not None:
                stage_store.close()
            shutil.rmtree(stage_dir, ignore_errors=True)

    def materialize_bound_assets(self, project: OcrProject) -> None:
        """Move newly imported runtime assets into an already bound project.

        Normal saves call this before SQLite writes.  Existing project assets
        are left in place; a newly imported page or layout artifact is copied
        into the user's sibling asset directory first.
        """
        if not project.db_path:
            return
        target = self._normalize_path(project.db_path)
        asset_root = target.parent / self.asset_dir_name(target)
        for page in project.pages:
            image = self._resolve_source_path(project, page.display_image_path)
            image_target = asset_root / "images" / f"{page.uid}{image.suffix or '.bin'}"
            self._copy_asset_in_place(image, image_target)
            page.image_path = str(image_target)
            page.cache_image_path = str(image_target)

            if page.thumbnail_path:
                thumbnail = self._resolve_source_path(project, page.thumbnail_path)
                thumbnail_target = asset_root / "thumbnails" / f"{page.uid}{thumbnail.suffix or '.bin'}"
                self._copy_asset_in_place(thumbnail, thumbnail_target)
                page.thumbnail_path = str(thumbnail_target)

            artifact = page.raw_layout_artifact
            if artifact is not None and artifact.artifact_path:
                source = self._resolve_source_path(project, artifact.artifact_path)
                artifact_target = asset_root / "artifacts" / f"{artifact.uid}{source.suffix or '.bin'}"
                self._copy_asset_in_place(source, artifact_target)
                artifact.artifact_path = str(artifact_target)

    @staticmethod
    def asset_dir_name(project_path: str | Path) -> str:
        return f"{Path(project_path).stem}.assets"

    @staticmethod
    def _normalize_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.suffix.lower() != ".ocrproj":
            path = path.with_suffix(".ocrproj")
        return path.resolve()

    @staticmethod
    def _load_single_project(store: ProjectStore) -> OcrProject:
        projects = store.list_projects()
        if len(projects) != 1:
            raise ProjectDataError(
                f"project file must contain exactly one project, found {len(projects)}"
            )
        project = store.load_project(project_id=int(projects[0]["id"]))
        if project is None:
            raise ProjectDataError("project file has no readable project")
        return project

    @staticmethod
    def _require_assets(project: OcrProject) -> None:
        for page in project.pages:
            image = Path(page.display_image_path)
            if not image.is_file():
                raise ProjectDataError(f"project page image is missing: {image}")

    def _materialize_assets_to_stage(
        self,
        project: OcrProject,
        stage_assets: Path,
        asset_dir_name: str,
    ) -> None:
        stage_assets.mkdir(parents=True)
        for page in project.pages:
            image = self._resolve_source_path(project, page.display_image_path)
            image_relative = self._copy_asset(
                image, stage_assets, asset_dir_name, "images", page.uid,
            )
            page.image_path = image_relative
            page.cache_image_path = image_relative
            if page.thumbnail_path:
                page.thumbnail_path = self._copy_asset(
                    self._resolve_source_path(project, page.thumbnail_path),
                    stage_assets,
                    asset_dir_name,
                    "thumbnails",
                    page.uid,
                )
            artifact = page.raw_layout_artifact
            if artifact is not None and artifact.artifact_path:
                artifact.artifact_path = self._copy_asset(
                    self._resolve_source_path(project, artifact.artifact_path),
                    stage_assets,
                    asset_dir_name,
                    "artifacts",
                    artifact.uid,
                )

    @staticmethod
    def _copy_asset(
        source: Path,
        stage_assets: Path,
        asset_dir_name: str,
        category: str,
        stable_name: str,
    ) -> str:
        if not source.is_file():
            raise ProjectDataError(f"project asset is missing: {source}")
        suffix = source.suffix or ".bin"
        relative = Path(asset_dir_name) / category / f"{stable_name}{suffix}"
        destination = stage_assets.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return relative.as_posix()

    @staticmethod
    def _copy_asset_in_place(source: Path, target: Path) -> None:
        if not source.is_file():
            raise ProjectDataError(f"project asset is missing: {source}")
        if source.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.copy-{uuid.uuid4().hex}")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _resolve_source_path(project: OcrProject, value: str) -> Path:
        path = Path(str(value or ""))
        if path.is_absolute() or not project.db_path:
            return path
        return Path(project.db_path).resolve().parent / path

    @staticmethod
    def _write_quality_probe_sidecar(stage_db: Path) -> None:
        store = qp.get_active_store()
        if store is not None:
            qp.save_store_to_path(store, str(stage_db) + ".qprobe.json")

    def _validate_stage(self, stage_db: Path) -> None:
        store = ProjectStore(str(stage_db))
        try:
            store.open()
            self._require_assets(self._load_single_project(store))
        finally:
            store.close()

    def _install_stage(self, stage_dir: Path, target: Path) -> None:
        stage_db = stage_dir / target.name
        stage_assets = stage_dir / self.asset_dir_name(target)
        stage_sidecar = Path(str(stage_db) + ".qprobe.json")
        target_assets = target.parent / self.asset_dir_name(target)
        target_sidecar = Path(str(target) + ".qprobe.json")
        token = uuid.uuid4().hex
        backups = {
            target: target.parent / f".{target.name}.backup-{token}",
            target_assets: target.parent / f".{target_assets.name}.backup-{token}",
            target_sidecar: target.parent / f".{target_sidecar.name}.backup-{token}",
        }
        moved: list[Path] = []
        installed: list[Path] = []
        try:
            for original, backup in backups.items():
                if original.exists():
                    os.replace(original, backup)
                    moved.append(original)
            os.replace(stage_assets, target_assets)
            installed.append(target_assets)
            if stage_sidecar.exists():
                os.replace(stage_sidecar, target_sidecar)
                installed.append(target_sidecar)
            os.replace(stage_db, target)
            installed.append(target)
        except Exception:
            for item in reversed(installed):
                self._remove_path(item)
            for original in reversed(moved):
                backup = backups[original]
                if backup.exists():
                    os.replace(backup, original)
            raise
        else:
            for backup in backups.values():
                self._remove_path(backup)

    def _activate_bound_paths(self, project: OcrProject, target: Path) -> None:
        root = target.parent
        for page in project.pages:
            page.image_path = str(root / page.image_path)
            page.cache_image_path = str(root / page.cache_image_path)
            if page.thumbnail_path:
                page.thumbnail_path = str(root / page.thumbnail_path)
            artifact = page.raw_layout_artifact
            if artifact is not None and artifact.artifact_path:
                artifact.artifact_path = str(root / artifact.artifact_path)

    def _capture_mutable_state(self, project: OcrProject) -> list[tuple[object, dict[str, Any]]]:
        state: list[tuple[object, dict[str, Any]]] = []
        seen: set[int] = set()

        def capture(obj: object, *names: str) -> None:
            if id(obj) in seen:
                return
            seen.add(id(obj))
            state.append((obj, {name: getattr(obj, name) for name in names}))

        capture(project, "id", "db_path")
        for page in project.pages:
            capture(page, "id", "image_path", "cache_image_path", "thumbnail_path")
            if page.raw_layout_artifact is not None:
                capture(page.raw_layout_artifact, "id", "artifact_path")
            for event in page.layout_edit_events:
                capture(event, "id")
            for block in self._blocks_for_page(page):
                capture(block, "id")
                for line in self._lines_for_block(block):
                    capture(line, "id")
                    for char in self._chars_for_line(line):
                        capture(char, "id")
        return state

    @staticmethod
    def _restore_mutable_state(state: list[tuple[object, dict[str, Any]]]) -> None:
        for obj, values in state:
            for name, value in values.items():
                setattr(obj, name, value)

    def _clear_storage_ids(self, project: OcrProject) -> None:
        project.id = None
        for page in project.pages:
            page.id = None
            if page.raw_layout_artifact is not None:
                page.raw_layout_artifact.id = None
            for event in page.layout_edit_events:
                event.id = None
            for block in self._blocks_for_page(page):
                block.id = None
                for line in self._lines_for_block(block):
                    line.id = None
                    for char in self._chars_for_line(line):
                        char.id = None

    @staticmethod
    def _blocks_for_page(page: Page) -> list[object]:
        if layout_snapshot_for_page(page) is None:
            return list(page.blocks)
        return [
            view.runtime_block
            for view in iter_page_layout_block_views(page)
            if view.runtime_block is not None
        ]

    @staticmethod
    def _lines_for_block(block: object) -> list[object]:
        values: list[object] = []
        seen: set[int] = set()
        for line in [*list(getattr(block, "lines", []) or []), *block_ocr_line_observations_by_uid(getattr(block, "uid", ""))]:
            if id(line) not in seen:
                seen.add(id(line))
                values.append(line)
        return values

    @staticmethod
    def _chars_for_line(line: object) -> list[object]:
        values: list[object] = []
        seen: set[int] = set()
        for char in [*list(getattr(line, "chars", []) or []), *line_ocr_chars_by_uid(getattr(line, "uid", ""))]:
            if id(char) not in seen:
                seen.add(id(char))
                values.append(char)
        return values

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


__all__ = ["BoundProject", "ProjectFileService"]
