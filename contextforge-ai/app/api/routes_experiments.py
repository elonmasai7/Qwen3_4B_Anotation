"""Experiment tracking APIs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ExperimentRecord(BaseModel):
    experiment_id: str
    model: str = "Qwen3-4B"
    retrieval_method: str = "hybrid"
    chunk_size: int = 2048
    overlap: int = 256
    examples: int = 6
    temperature: float = 0.1
    score: float = 0.0
    latency: float = 0.0


EXPERIMENT_LOG = Path("experiments/results/experiments.jsonl")


@router.post("/log")
def log_experiment(record: ExperimentRecord) -> dict:
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump()
    payload["created_at"] = datetime.now(UTC).isoformat()
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return {"status": "saved", "path": str(EXPERIMENT_LOG)}


@router.get("/list")
def list_experiments() -> dict:
    if not EXPERIMENT_LOG.exists():
        return {"experiments": []}
    rows = [json.loads(line) for line in EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"experiments": rows[-200:]}
