from pathlib import Path


def test_install_fonts_loads_bundled_files_without_external_zip(tmp_path, monkeypatch):
    from app.utils import font_installer

    fonts_dir = tmp_path / "resources" / "fonts"
    fonts_dir.mkdir(parents=True)
    (fonts_dir / "JetBrainsMono-Regular.ttf").write_bytes(b"fake-ttf")
    (fonts_dir / "Custom.otf").write_bytes(b"fake-otf")

    loaded: list[str] = []

    def fake_add_application_font(path: str) -> int:
        loaded.append(Path(path).name)
        return len(loaded)

    monkeypatch.setattr(font_installer.QFontDatabase, "addApplicationFont", fake_add_application_font)
    monkeypatch.setattr(font_installer.QFontDatabase, "applicationFontFamilies", lambda _font_id: ["Fake"])

    font_installer.install_fonts(tmp_path)

    assert sorted(loaded) == ["Custom.otf", "JetBrainsMono-Regular.ttf"]
