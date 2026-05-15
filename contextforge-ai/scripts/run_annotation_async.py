"""Queue-based async annotation runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.dataset_loader import DatasetLoader
from app.core.exporter import Exporter
from app.core.job_queue import AnnotationJobQueue
from app.db.database import init_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    args = parser.parse_args()

    init_db()
    loader = DatasetLoader()
    queue = AnnotationJobQueue()
    exporter = Exporter()

    samples = loader.load(args.input)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    job_ids = [queue.enqueue(sample) for sample in samples]
    start = time.time()
    completed: dict[str, dict] = {}
    failed: dict[str, str] = {}

    while len(completed) + len(failed) < len(job_ids):
        if time.time() - start > args.wait_timeout:
            raise TimeoutError("Timed out waiting for async annotation jobs")
        time.sleep(0.2)
        for job_id in job_ids:
            if job_id in completed or job_id in failed:
                continue
            status = queue.get_job(job_id)
            if status is None:
                failed[job_id] = "missing"
                continue
            if status["status"] == "completed":
                completed[job_id] = status["result"]
            elif status["status"] == "failed":
                failed[job_id] = status.get("error", "unknown")

    predictions = [
        {
            "id": record["prediction"]["id"],
            "label": record["prediction"]["label"],
            "confidence": record["prediction"]["confidence"],
        }
        for record in completed.values()
    ]
    exporter.export_predictions_jsonl(predictions, args.output)
    print(
        json.dumps(
            {
                "queued": len(job_ids),
                "completed": len(completed),
                "failed": len(failed),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
