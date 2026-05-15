from pathlib import Path

from app.core.exporter import Exporter


def test_exporter_jsonl(tmp_path: Path) -> None:
    out_path = tmp_path / "preds.jsonl"
    rows = [{"id": "1", "label": "x", "confidence": 0.7}]
    Exporter().export_predictions_jsonl(rows, str(out_path))
    assert out_path.exists()
    assert "\"id\": \"1\"" in out_path.read_text(encoding="utf-8")
