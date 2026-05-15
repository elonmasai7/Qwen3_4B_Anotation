"""Annotation APIs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.annotation_pipeline import AnnotationPipeline
from app.core.dataset_loader import DatasetLoader
from app.core.exporter import Exporter
from app.core.job_queue import AnnotationJobQueue
from app.db.database import init_db

router = APIRouter()
pipeline = AnnotationPipeline()
loader = DatasetLoader()
exporter = Exporter()
job_queue = AnnotationJobQueue()


class AnnotationRunRequest(BaseModel):
    input_path: str
    output_path: str
    max_samples: int | None = None


@router.post("/run")
def run_annotation(req: AnnotationRunRequest) -> dict:
    in_path = Path(req.input_path)
    if not in_path.exists():
        raise HTTPException(status_code=404, detail="Input dataset file not found")

    samples = loader.load(str(in_path))
    if req.max_samples:
        samples = samples[: req.max_samples]

    example_pool: list[dict] = []
    predictions: list[dict] = []
    latencies: list[float] = []
    for sample in samples:
        pred, latency = pipeline.annotate_sample(sample, example_pool)
        predictions.append(pred)
        latencies.append(latency)
        example_pool.append(
            {
                "input_text": sample["input_text"],
                "label": pred["label"],
            }
        )

    exporter.export_predictions_jsonl(predictions, req.output_path)
    return {
        "total": len(predictions),
        "avg_latency": sum(latencies) / max(len(latencies), 1),
        "output_path": req.output_path,
    }


class AnnotationSingleRequest(BaseModel):
    sample: dict
    example_pool: list[dict] = Field(default_factory=list)


@router.post("/single")
def run_single_annotation(req: AnnotationSingleRequest) -> dict:
    sample = req.sample
    if "id" not in sample or "input_text" not in sample:
        raise HTTPException(status_code=422, detail="Sample must include id and input_text")
    pred, latency = pipeline.annotate_sample(sample, req.example_pool)
    return {"prediction": pred, "latency": latency}


class AnnotationJobRequest(BaseModel):
    sample: dict
    example_pool: list[dict] = Field(default_factory=list)


@router.post("/enqueue")
def enqueue_annotation(req: AnnotationJobRequest) -> dict:
    if "id" not in req.sample or "input_text" not in req.sample:
        raise HTTPException(status_code=422, detail="Sample must include id and input_text")
    init_db()
    job_id = job_queue.enqueue(req.sample, req.example_pool)
    return {"job_id": job_id, "status": "queued"}


@router.get("/job/{job_id}")
def get_annotation_job(job_id: str) -> dict:
    init_db()
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
