"""Dataset APIs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dataset_loader import DatasetLoader

router = APIRouter()
loader = DatasetLoader()


class DatasetLoadRequest(BaseModel):
    path: str


@router.post("/load")
def load_dataset(req: DatasetLoadRequest) -> dict:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    rows = loader.load(str(path))
    return {"rows": rows[:20], "total_rows": len(rows)}
