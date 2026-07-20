"""User-selected project files and their sibling asset directories."""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import uuid

from app.infrastructure.project_store import ProjectStore, ProjectStoreDataError
from app.models.project_session import PageRecord, ProjectSession


@dataclass(frozen=True)
class BoundProject:
    """A project session and the strict store for its selected file."""

    store: ProjectStore
    session: ProjectSession


class ProjectFileService:
    """Open and atomically persist project-session files with their assets."""

    def open_project(self, file_path: str | Path) -> BoundProject:
        target = self._normalize_path(file_path)
        if not target.is_file():
            raise ProjectStoreDataError(f"project file does not exist: {target}")

        store = ProjectStore(target)
        persisted = store.load_session()
        self._validate_page_assets(persisted, target)
        session = self._with_resolved_page_paths(persisted, target)
        return BoundProject(store=store, session=session)

    def bind_project(
        self,
        value: ProjectSession | BoundProject,
        file_path: str | Path,
    ) -> BoundProject:
        """Create or rebind a project file without mutating ``value``."""
        session = self._session_from(value)
        return self._persist_staged(session, self._normalize_path(file_path))

    def save_as(
        self,
        value: ProjectSession | BoundProject,
        file_path: str | Path,
    ) -> BoundProject:
        """Persist the session to a new selected path and return its binding."""
        return self.bind_project(value, file_path)

    def save_project(self, value: ProjectSession | BoundProject) -> BoundProject:
        """Materialize new assets and persist a session to its bound path."""
        session = self._session_from(value)
        target = self._bound_path(session)
        if isinstance(value, BoundProject):
            store_target = self._normalize_path(value.store.path)
            if store_target != target:
                raise ProjectStoreDataError(
                    "bound store path does not match the session save path"
                )
        return self._persist_staged(session, target)

    def save_session(self, value: ProjectSession | BoundProject) -> BoundProject:
        """Persist a v2 session; this is the explicit session-named save API."""
        return self.save_project(value)

    def materialize_bound_assets(
        self,
        value: ProjectSession | BoundProject,
    ) -> BoundProject:
        """Run the normal bound save, including asset materialization."""
        return self.save_project(value)

    @staticmethod
    def asset_dir_name(project_path: str | Path) -> str:
        return f"{Path(project_path).stem}.assets"

    def _persist_staged(self, session: ProjectSession, target: Path) -> BoundProject:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = target.parent / f".{target.stem}.save-stage-{uuid.uuid4().hex}"
        stage_dir.mkdir()
        stage_db = stage_dir / target.name
        stage_assets = stage_dir / self.asset_dir_name(target)

        try:
            staged_session = self._stage_assets(session, target, stage_assets)
            ProjectStore(stage_db).save_session(staged_session)
            self._validate_stage(stage_db)
            self._install_stage(stage_dir, target)

            active_session = self._with_resolved_page_paths(staged_session, target)
            return BoundProject(
                store=ProjectStore(target),
                session=active_session,
            )
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def _stage_assets(
        self,
        session: ProjectSession,
        target: Path,
        stage_assets: Path,
    ) -> ProjectSession:
        stage_assets.mkdir(parents=True)
        pages: list[PageRecord] = []
        asset_dir = self.asset_dir_name(target)
        for page in session.page_repository.all():
            image_source = self._resolve_input_path(session, page.image_path)
            image_path = self._copy_asset(
                image_source,
                stage_assets,
                Path(asset_dir) / "images" / self._asset_name(page.uid, image_source),
                page.uid,
                "image_path",
            )

            cache_path = ""
            if page.cache_image_path:
                cache_source = self._resolve_input_path(session, page.cache_image_path)
                if cache_source.resolve() == image_source.resolve():
                    cache_path = image_path
                else:
                    cache_path = self._copy_asset(
                        cache_source,
                        stage_assets,
                        Path(asset_dir)
                        / "images"
                        / self._asset_name(page.uid + ".cache", cache_source),
                        page.uid,
                        "cache_image_path",
                    )

            thumbnail_path = ""
            if page.thumbnail_path:
                thumbnail_source = self._resolve_input_path(session, page.thumbnail_path)
                thumbnail_path = self._copy_asset(
                    thumbnail_source,
                    stage_assets,
                    Path(asset_dir)
                    / "thumbnails"
                    / self._asset_name(page.uid, thumbnail_source),
                    page.uid,
                    "thumbnail_path",
                )

            pages.append(
                replace(
                    page,
                    image_path=image_path,
                    cache_image_path=cache_path,
                    thumbnail_path=thumbnail_path,
                )
            )

        return self._with_pages(session, tuple(pages), save_path=None)

    @staticmethod
    def _asset_name(stable_uid: str, source: Path) -> str:
        return f"{stable_uid}{source.suffix or '.bin'}"

    @staticmethod
    def _copy_asset(
        source: Path,
        stage_assets: Path,
        relative: Path,
        page_uid: str,
        field_name: str,
    ) -> str:
        if not source.is_file():
            raise ProjectStoreDataError(
                f"page {page_uid!r} {field_name} asset is missing: {source}"
            )
        destination = stage_assets.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return relative.as_posix()

    def _validate_stage(self, stage_db: Path) -> None:
        staged = ProjectStore(stage_db).load_session()
        self._validate_page_assets(staged, stage_db)

    def _install_stage(self, stage_dir: Path, target: Path) -> None:
        stage_db = stage_dir / target.name
        stage_assets = stage_dir / self.asset_dir_name(target)
        target_assets = target.parent / self.asset_dir_name(target)
        token = uuid.uuid4().hex
        originals = (target, target_assets)
        backups = {
            original: target.parent / f".{original.name}.backup-{token}"
            for original in originals
        }
        moved: list[Path] = []
        installed: list[Path] = []

        try:
            for original in originals:
                backup = backups[original]
                if original.exists():
                    os.replace(original, backup)
                    moved.append(original)

            os.replace(stage_assets, target_assets)
            installed.append(target_assets)
            os.replace(stage_db, target)
            installed.append(target)
        except Exception:
            for path in reversed(installed):
                self._remove_path(path)
            for original in reversed(moved):
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

    def _validate_page_assets(
        self,
        session: ProjectSession,
        project_path: Path,
    ) -> None:
        for page in session.page_repository.all():
            self._require_asset(page, "image_path", page.image_path, project_path)
            if page.cache_image_path:
                self._require_asset(
                    page,
                    "cache_image_path",
                    page.cache_image_path,
                    project_path,
                )
            if page.thumbnail_path:
                self._require_asset(
                    page,
                    "thumbnail_path",
                    page.thumbnail_path,
                    project_path,
                )

    @staticmethod
    def _require_asset(
        page: PageRecord,
        field_name: str,
        value: str,
        project_path: Path,
    ) -> None:
        path = ProjectFileService._resolve_path(value, project_path)
        if not path.is_file():
            raise ProjectStoreDataError(
                f"page {page.uid!r} {field_name} asset is missing: {path}"
            )

    def _with_resolved_page_paths(
        self,
        session: ProjectSession,
        project_path: Path,
    ) -> ProjectSession:
        pages = tuple(
            replace(
                page,
                image_path=str(self._resolve_path(page.image_path, project_path)),
                cache_image_path=(
                    str(self._resolve_path(page.cache_image_path, project_path))
                    if page.cache_image_path
                    else ""
                ),
                thumbnail_path=(
                    str(self._resolve_path(page.thumbnail_path, project_path))
                    if page.thumbnail_path
                    else ""
                ),
            )
            for page in session.page_repository.all()
        )
        return self._with_pages(session, pages, save_path=str(project_path))

    @staticmethod
    def _resolve_path(value: str, project_path: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        return (project_path.parent / path).resolve()

    @staticmethod
    def _resolve_input_path(session: ProjectSession, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        if session.save_path:
            base = Path(session.save_path).expanduser().resolve().parent
        else:
            base = Path.cwd()
        return (base / path).resolve()

    @staticmethod
    def _with_pages(
        session: ProjectSession,
        pages: tuple[PageRecord, ...],
        *,
        save_path: str | None,
    ) -> ProjectSession:
        return ProjectSession.from_records(
            session.project_record,
            pages=pages,
            paddle_artifacts=session.paddle_artifact_repository.all(),
            layouts=session.layout_repository.all(),
            ocr_runs=session.ocr_observation_repository.all_runs(),
            ocr_regions=session.ocr_observation_repository.all_regions(),
            ocr_lines=session.ocr_observation_repository.all_lines(),
            ocr_atoms=session.ocr_observation_repository.all_atoms(),
            ocr_candidates=session.ocr_observation_repository.all_candidates(),
            ocr_batches=session.ocr_observation_repository.all_batches(),
            ocr_active_pointers=session.ocr_observation_repository.all_active_pointers(),
            proof_states=session.proof_repository.all_states(),
            bindings=session.binding_repository.all(),
            table_texts=session.table_text_repository.all(),
            save_path=save_path,
        )

    @staticmethod
    def _normalize_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.suffix.lower() != ".ocrproj":
            path = path.with_suffix(".ocrproj")
        return path.resolve()

    def _bound_path(self, session: ProjectSession) -> Path:
        if not session.save_path:
            raise ProjectStoreDataError("project session has no bound save path")
        return self._normalize_path(session.save_path)

    @staticmethod
    def _session_from(value: ProjectSession | BoundProject) -> ProjectSession:
        if isinstance(value, BoundProject):
            return value.session
        if isinstance(value, ProjectSession):
            return value
        raise TypeError("project file service requires ProjectSession or BoundProject")


__all__ = ["BoundProject", "ProjectFileService"]
