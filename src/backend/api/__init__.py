from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
import uuid

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


@app.get("/")
async def root():
    return {"message": "Annotation Platform API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest):
    annotation_id = str(uuid.uuid4())

    return AnnotationResponse(
        annotation_id=annotation_id,
        status="completed",
        labels=[{"value": "sample_label", "confidence": 0.85}],
        processing_time_ms=150.0,
    )


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    dataset_id = str(uuid.uuid4())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        file_count=1,
        status="uploaded",
    )


@app.get("/datasets")
async def list_datasets():
    return {"datasets": []}


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    return {"dataset_id": dataset_id, "name": "sample", "status": "ready"}


@app.post("/experiments", response_model=dict)
async def create_experiment(request: ExperimentCreateRequest):
    experiment_id = str(uuid.uuid4())

    return {
        "experiment_id": experiment_id,
        "name": request.name,
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/experiments")
async def list_experiments():
    return {"experiments": []}


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    return {
        "experiment_id": experiment_id,
        "name": "sample_experiment",
        "status": "completed",
        "metrics": {
            "accuracy": 0.85,
            "f1": 0.82,
            "latency_ms": 150.0,
        },
    }


@app.get("/leaderboard")
async def get_leaderboard():
    return {
        "rankings": [
            {"rank": 1, "experiment_id": "exp_001", "score": 0.92, "name": "Baseline v1"},
            {"rank": 2, "experiment_id": "exp_002", "score": 0.89, "name": "Optimized v2"},
        ]
    }


@app.post("/leaderboard/submit")
async def submit_leaderboard(experiment_id: str):
    return {
        "submission_id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "status": "submitted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prompts")
async def list_prompts():
    return {"prompts": []}


@app.post("/prompts")
async def create_prompt(prompt: dict[str, Any]):
    prompt_id = str(uuid.uuid4())
    return {"prompt_id": prompt_id, **prompt}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)