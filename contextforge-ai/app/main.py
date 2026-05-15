"""FastAPI entrypoint for ContextForge AI."""

from fastapi import FastAPI

from app.api.routes_annotation import router as annotation_router
from app.api.routes_dataset import router as dataset_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_experiments import router as experiments_router
from app.config import settings
from app.db.database import init_db

app = FastAPI(title=settings.app_name, version="1.0.0")

app.include_router(dataset_router, prefix="/dataset", tags=["dataset"])
app.include_router(annotation_router, prefix="/annotation", tags=["annotation"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
app.include_router(experiments_router, prefix="/experiments", tags=["experiments"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}
