from pathlib import Path

from app.core.dataset_loader import DatasetLoader


def test_dataset_loader_jsonl(tmp_path: Path) -> None:
    payload = (
        '{"id":"1","input_text":"Doc A","task_instruction":"Classify","label_options":"yes|no"}\n'
        '{"id":"2","input_text":"Doc B","task_instruction":"Classify","label_options":"yes|no"}\n'
    )
    data_file = tmp_path / "sample.jsonl"
    data_file.write_text(payload, encoding="utf-8")

    rows = DatasetLoader().load(str(data_file))
    assert len(rows) == 2
    assert rows[0]["id"] == "1"
    assert rows[0]["label_options"] == ["yes", "no"]
