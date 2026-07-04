from app.core import quality_probe as qp
from app.core.proof_line_mutation import set_line_proof_text
from app.models import BBox, Block, BlockType, Char, Line, OcrProject, Page
from app.services.project_diagnostics import diagnose_project


def _project_with_line(line: Line) -> OcrProject:
    block = Block(block_type=BlockType.TEXT, bbox=BBox(0, 0, 100, 20), lines=[line])
    page = Page(image_path="/tmp/diagnostics.png", width=100, height=100, blocks=[block])
    return OcrProject(name="diag", pages=[page])


def test_project_diagnostics_reports_text_char_mismatch_and_duplicate_uid():
    line = Line(
        text="甲乙",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=BBox(0, 0, 20, 20), uid="char_dup"),
            Char(char="丙", confidence=0.9, bbox=BBox(20, 0, 20, 20), uid="char_dup"),
        ],
    )
    set_line_proof_text(line, "甲乙")
    report = diagnose_project(_project_with_line(line))

    codes = report.counts_by_code()
    assert codes["line_chars_text_mismatch"] == 1
    assert codes["duplicate_char_uid"] == 1
    assert report.error_count == 2


def test_project_diagnostics_reports_missing_and_invalid_char_bbox():
    line = Line(
        text="甲乙",
        confidence=0.9,
        bbox=BBox(0, 0, 40, 20),
        chars=[
            Char(char="甲", confidence=0.9, bbox=None),
            Char(char="乙", confidence=0.9, bbox=BBox(20, 0, 0, 20)),
        ],
    )
    report = diagnose_project(_project_with_line(line))

    codes = report.counts_by_code()
    assert codes["char_missing_bbox"] == 1
    assert codes["char_invalid_bbox"] == 1


def test_project_diagnostics_reports_stale_pending_probe_anchor():
    line = Line(
        text="已",
        confidence=0.9,
        bbox=BBox(0, 0, 20, 20),
        chars=[Char(char="已", confidence=0.9, bbox=BBox(0, 0, 20, 20))],
    )
    project = _project_with_line(line)
    page = project.pages[0]
    probe_store = qp.ProbeStore()
    probe_store.add(
        qp.Probe(
            key=qp.ProbeKey(
                page_number=page.page_number,
                block_index=0,
                line_index=0,
                char_index=0,
            ),
            true_char="已",
            fake_char="己",
        )
    )
    set_line_proof_text(line, "巳")

    report = diagnose_project(project, probe_store=probe_store)

    assert report.counts_by_code()["stale_pending_probe_anchor"] == 1


def test_project_diagnostics_script_help():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_project.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Diagnose proof/OCR text and geometry drift" in result.stdout
    assert "--fail-on" in result.stdout
