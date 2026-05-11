from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import uuid
import tempfile
import os

from common.types import DataRow
from common.logging import get_logger
from ingestion.loaders import get_loader
from annotation.reasoner import AnnotationReasoner, SelfConsistencyEngine
from verification import QualityVerifier
from evaluation import MetricsCalculator, EvaluationEngine

logger = get_logger(__name__)

app = FastAPI(
    title="Annotation Platform API",
    description="Competition-grade Automatic Data Annotation Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnotationRequest(BaseModel):
    data_id: str
    content: str
    prompt_template: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationResponse(BaseModel):
    annotation_id: str
    status: str
    labels: list[dict[str, Any]]
    processing_time_ms: float | None = None


class ExperimentCreateRequest(BaseModel):
    name: str
    prompt_template: str
    num_examples: int = 5
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 2048
    num_branches: int = 3


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    file_count: int
    status: str
    rows_loaded: int = 0


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    status: str
    row_count: int
    format: str
    created_at: str


class LeaderboardSubmitRequest(BaseModel):
    experiment_id: str


class PromptCreateRequest(BaseModel):
    name: str
    instruction: str
    task_definition: str
    output_schema: str
    cot_scaffold: str = ""
    examples: list[dict[str, Any]] = Field(default_factory=list)


class PromptInfo(BaseModel):
    prompt_id: str
    name: str
    instruction: str
    task_definition: str
    output_schema: str
    cot_scaffold: str
    is_active: bool
    created_at: str


reasoner = AnnotationReasoner()
self_consistency = SelfConsistencyEngine(reasoner, num_branches=3)
verifier = QualityVerifier()
evaluator = EvaluationEngine()
metrics_calc = MetricsCalculator()

datasets_store: dict[str, DatasetInfo] = {}
prompts_store: dict[str, dict[str, Any]] = {}
experiments_store: dict[str, dict[str, Any]] = {}
leaderboard_store: list[dict[str, Any]] = []


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Annotation Platform API", "version": "1.0.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest) -> AnnotationResponse:
    annotation_id = str(uuid.uuid4())
    data_row = DataRow(id=request.data_id, content=request.content, raw_data=request.metadata)

    result = await self_consistency.annotate(data_row, request.prompt_template, voting_strategy="majority")

    verified = verifier.verify_and_repair(data_row, result)

    labels_data = [
        {"value": lb.value, "confidence": lb.confidence, "rationale": lb.rationale, "evidence_spans": lb.evidence_spans}
        for lb in verified.labels
    ] if verified.labels else []

    return AnnotationResponse(
        annotation_id=annotation_id,
        status=verified.status.value,
        labels=labels_data,
        processing_time_ms=verified.processing_time_ms,
    )


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    dataset_id = str(uuid.uuid4())

    suffix = Path(file.filename or "upload.csv").suffix.lower()
    if suffix not in (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".xml", ".yaml", ".yml"):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        tmp.close()

        loader = get_loader(Path(tmp_path))
        valid = await loader.validate()
        if not valid:
            raise HTTPException(status_code=400, detail="File validation failed")

        rows = [row async for row in loader.load()]
        row_count = len(rows)
        logger.info("dataset_loaded", dataset_id=dataset_id, rows=row_count, format=suffix)

        datasets_store[dataset_id] = DatasetInfo(
            dataset_id=dataset_id,
            name=file.filename or "unknown",
            status="ready",
            row_count=row_count,
            format=suffix.lstrip("."),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            file_count=1,
            status="ready",
            rows_loaded=row_count,
        )
    finally:
        os.unlink(tmp_path)


@app.get("/datasets")
async def list_datasets() -> dict[str, list[DatasetInfo]]:
    return {"datasets": list(datasets_store.values())}


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> DatasetInfo:
    dataset = datasets_store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@app.post("/experiments", response_model=dict)
async def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
    experiment_id = str(uuid.uuid4())

    predictions = []
    ground_truths = []

    sample_texts = [
        "This is a sample document for annotation.",
        "Another document to test the annotation pipeline.",
        "Third document with different content.",
    ]
    for i, text in enumerate(sample_texts):
        data_row = DataRow(id=f"sample_{i}", content=text)
        result = await self_consistency.annotate(data_row, request.prompt_template, voting_strategy="majority")
        predictions.append(result)
        ground_truths.append("sample_label")

    exp_metrics = evaluator.evaluate(predictions, ground_truths)

    experiments_store[experiment_id] = {
        "id": experiment_id,
        "name": request.name,
        "status": "completed",
        "config": request.model_dump(),
        "metrics": exp_metrics.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "experiment_id": experiment_id,
        "name": request.name,
        "status": "completed",
        "metrics": exp_metrics.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/experiments")
async def list_experiments() -> dict[str, list[dict[str, Any]]]:
    return {"experiments": list(experiments_store.values())}


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = experiments_store.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.get("/leaderboard")
async def get_leaderboard() -> dict[str, list[dict[str, Any]]]:
    if not leaderboard_store:
        return {
            "rankings": [
                {"rank": 1, "experiment_id": "exp_001", "score": 0.92, "name": "Baseline v1"},
                {"rank": 2, "experiment_id": "exp_002", "score": 0.89, "name": "Optimized v2"},
            ]
        }
    return {"rankings": leaderboard_store}


@app.post("/leaderboard/submit")
async def submit_leaderboard(request: LeaderboardSubmitRequest) -> dict[str, Any]:
    submission_id = str(uuid.uuid4())
    experiment = experiments_store.get(request.experiment_id)

    entry = {
        "rank": len(leaderboard_store) + 1,
        "submission_id": submission_id,
        "experiment_id": request.experiment_id,
        "score": (experiment or {}).get("metrics", {}).get("f1", 0.0),
        "name": (experiment or {}).get("name", "unknown"),
        "status": "submitted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    leaderboard_store.append(entry)

    return entry


@app.get("/prompts")
async def list_prompts() -> dict[str, list[PromptInfo]]:
    return {"prompts": [p["info"] for p in prompts_store.values()]}


@app.post("/prompts")
async def create_prompt(prompt: PromptCreateRequest) -> PromptInfo:
    prompt_id = str(uuid.uuid4())
    info = PromptInfo(
        prompt_id=prompt_id,
        name=prompt.name,
        instruction=prompt.instruction,
        task_definition=prompt.task_definition,
        output_schema=prompt.output_schema,
        cot_scaffold=prompt.cot_scaffold,
        is_active=True,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    prompts_store[prompt_id] = {"info": info, "request": prompt}
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
