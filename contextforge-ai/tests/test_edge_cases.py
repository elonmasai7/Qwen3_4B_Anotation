import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes_annotation import router as annotation_router
from app.core.dataset_loader import DatasetLoader
from app.core.qwen_inference import QwenInference
from app.core.repair_agent import RepairAgent
from app.core.schema_detector import SchemaDetector
from app.core.verifier_agent import VerifierAgent
from app.db.database import get_db
from app.main import app


def test_dataset_loader_csv_and_unsupported(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,text\n1,hello world\n", encoding="utf-8")
    rows = DatasetLoader().load(str(csv_file))
    assert rows[0]["id"] == "1"

    bad_file = tmp_path / "sample.txt"
    bad_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        DatasetLoader().load(str(bad_file))


def test_schema_detector_fallback_pick() -> None:
    schema = SchemaDetector().detect(["uid", "body_text", "instruction_text", "possible_labels"])
    assert schema.id_field == "uid"
    assert schema.text_field == "body_text"


def test_repair_and_verifier_paths() -> None:
    verifier = VerifierAgent()
    repair = RepairAgent()
    invalid = {"confidence": 0.1}
    check = verifier.verify(invalid, ["A", "B"])
    fixed = repair.repair(invalid, check["issues"], ["A", "B"])
    assert fixed["label"] in {"A", "B"}
    assert fixed["confidence"] >= 0.35


def test_qwen_inference_mock_and_transformer_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    inf = QwenInference()
    inf._backend = "mock"
    out = inf.generate_json("positive negative")
    assert "label" in out

    class FakeGen:
        def __call__(self, *_args, **_kwargs):
            return [{"generated_text": '{"label":"x","confidence":0.6,"evidence":["e"],"rationale":"r","uncertainty_reason":"u"}'}]

    inf._backend = "transformers"
    inf._generator = FakeGen()
    out2 = inf.generate_json("any")
    assert out2["label"] == "x"

    # malformed model output should fallback to mock
    class BadGen:
        def __call__(self, *_args, **_kwargs):
            return [{"generated_text": "not-json"}]

    inf._generator = BadGen()
    out3 = inf.generate_json("positive negative", retries=0)
    assert "label" in out3


def test_routes_annotation_error_branches(tmp_path: Path) -> None:
    client = TestClient(app)
    missing = client.post("/annotation/run", json={"input_path": "missing.jsonl", "output_path": "x.jsonl"})
    assert missing.status_code == 404

    bad_single = client.post("/annotation/single", json={"sample": {"foo": "bar"}})
    assert bad_single.status_code == 422


def test_get_db_generator() -> None:
    gen = get_db()
    db = next(gen)
    assert db is not None
    with pytest.raises(StopIteration):
        next(gen)
