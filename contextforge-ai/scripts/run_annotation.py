"""Run end-to-end annotation from CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.annotation_pipeline import AnnotationPipeline
from app.core.dataset_loader import DatasetLoader
from app.core.exporter import Exporter
from app.utils.metrics import RunMetrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    loader = DatasetLoader()
    pipeline = AnnotationPipeline()
    exporter = Exporter()
    metrics = RunMetrics()

    samples = loader.load(args.input)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    predictions: list[dict] = []
    example_pool: list[dict] = []
    for sample in samples:
        pred, _ = pipeline.annotate_sample(sample, example_pool)
        predictions.append(
            {
                "id": pred["id"],
                "label": pred["label"],
                "confidence": pred["confidence"],
            }
        )
        example_pool.append({"input_text": sample["input_text"], "label": pred["label"]})

    exporter.export_predictions_jsonl(predictions, args.output)
    metrics.finish(len(predictions))
    print(f"Annotated {len(predictions)} samples | throughput={metrics.throughput:.2f}/s")


if __name__ == "__main__":
    main()
