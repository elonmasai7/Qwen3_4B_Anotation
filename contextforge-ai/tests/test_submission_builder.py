from pathlib import Path

from app.core.exporter import Exporter


def test_build_submission_bundle(tmp_path: Path) -> None:
    report = tmp_path / "TECHNICAL_REPORT.md"
    report.write_text("# report", encoding="utf-8")

    src_root = tmp_path / "src"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "file.txt").write_text("hello", encoding="utf-8")

    out_dir = tmp_path / "submission"
    out = Exporter().build_submission_bundle(
        predictions=[{"id": "1", "label": "A", "confidence": 0.9}],
        config={"model": "Qwen3-4B"},
        technical_report_path=str(report),
        source_root=str(src_root),
        output_dir=str(out_dir),
    )
    assert Path(out).exists()
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "source_code.zip").exists()
