import logging
from pathlib import Path
from PySide6.QtGui import QFontDatabase

logger = logging.getLogger(__name__)

def install_fonts(project_root: Path):
    """Load bundled application fonts into ``QFontDatabase``."""
    fonts_dir = project_root / "resources" / "fonts"
    if not fonts_dir.exists():
        logger.debug("Bundled font directory not found: %s", fonts_dir)
        return

    font_files = sorted([
        *fonts_dir.glob("**/*.ttf"),
        *fonts_dir.glob("**/*.otf"),
    ])
    if not font_files:
        logger.debug("No bundled font files found in %s", fonts_dir)
        return

    for font_path in font_files:
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id == -1:
            logger.warning("Failed to load font: %s", font_path)
        else:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            logger.info("Loaded font: %s", font_families)
