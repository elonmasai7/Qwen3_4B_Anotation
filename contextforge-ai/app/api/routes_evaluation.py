"""Evaluation APIs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.evaluator import Evaluator

router = APIRouter()
evaluator = Evaluator()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class EvaluationRequest(BaseModel):
    pred_path: str
    gold_path: str


@router.post("/run")
def run_evaluation(req: EvaluationRequest) -> dict:
    pred_path = Path(req.pred_path)
    gold_path = Path(req.gold_path)
    if not pred_path.exists() or not gold_path.exists():
        raise HTTPException(status_code=404, detail="Prediction or gold file not found")

    pred_rows = _read_jsonl(pred_path)
    gold_rows = _read_jsonl(gold_path)
    report = evaluator.evaluate(pred_rows, gold_rows)
    evaluator.save_report(
        report,
        "data/reports/evaluation_report.json",
        "data/reports/evaluation_report.md",
    )
    return report
