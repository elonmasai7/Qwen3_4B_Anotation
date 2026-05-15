"""Build leaderboard submission package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.exporter import Exporter


def _load_predictions(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows.append(
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "confidence": float(obj.get("confidence", 0.0)),
                }
            )
    return rows


def main() -> None:
    pred_path = Path("data/predictions/preds.jsonl")
    if not pred_path.exists():
        raise FileNotFoundError("Expected predictions at data/predictions/preds.jsonl")

    predictions = _load_predictions(str(pred_path))
    exporter = Exporter()
    cfg = {
        "model": "Qwen3-4B",
        "retrieval_method": "hybrid",
        "chunk_size": 2048,
        "overlap": 256,
        "examples": 6,
        "temperature": 0.1,
    }
    out = exporter.build_submission_bundle(
        predictions=predictions,
        config=cfg,
        technical_report_path="TECHNICAL_REPORT.md",
        source_root=".",
        output_dir="submission",
    )
    print(f"Submission bundle ready at: {out}")


if __name__ == "__main__":
    main()
