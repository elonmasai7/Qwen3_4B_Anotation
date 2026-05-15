import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_health_route() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dataset_and_annotation_routes(tmp_path: Path) -> None:
    payload = {
        "id": "s1",
        "input_text": "A very positive outcome.",
        "task_instruction": "Classify sentiment",
        "label_options": ["positive", "negative"],
    }
    data_file = tmp_path / "input.jsonl"
    data_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    out_file = tmp_path / "preds.jsonl"

    client = TestClient(app)
    load_resp = client.post("/dataset/load", json={"path": str(data_file)})
    assert load_resp.status_code == 200
    assert load_resp.json()["total_rows"] == 1

    run_resp = client.post(
        "/annotation/run",
        json={
            "input_path": str(data_file),
            "output_path": str(out_file),
            "max_samples": 1,
        },
    )
    assert run_resp.status_code == 200
    assert out_file.exists()


def test_evaluation_and_experiment_routes(tmp_path: Path) -> None:
    pred = tmp_path / "pred.jsonl"
    gold = tmp_path / "gold.jsonl"
    pred.write_text('{"id":"1","label":"A","confidence":0.7}\n', encoding="utf-8")
    gold.write_text('{"id":"1","label":"A"}\n', encoding="utf-8")

    client = TestClient(app)
    eval_resp = client.post("/evaluation/run", json={"pred_path": str(pred), "gold_path": str(gold)})
    assert eval_resp.status_code == 200
    assert "accuracy" in eval_resp.json()

    exp_resp = client.post("/experiments/log", json={"experiment_id": "exp_test_001"})
    assert exp_resp.status_code == 200
    list_resp = client.get("/experiments/list")
    assert list_resp.status_code == 200


def test_async_job_routes() -> None:
    client = TestClient(app)
    payload = {
        "sample": {
            "id": "job_sample_001",
            "input_text": "A clear positive signal appears throughout the report.",
            "task_instruction": "Classify sentiment",
            "label_options": ["positive", "negative", "neutral"],
        }
    }
    enqueue = client.post("/annotation/enqueue", json=payload)
    assert enqueue.status_code == 200
    job_id = enqueue.json()["job_id"]

    status = client.get(f"/annotation/job/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running", "completed", "failed"}
