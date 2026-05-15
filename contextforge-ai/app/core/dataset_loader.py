"""Dataset loading, normalization, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.core.schema_detector import SchemaDetector
from app.utils.text_cleaning import normalize_text


class DatasetLoader:
    """Loads datasets into a standardized annotation schema."""

    def __init__(self) -> None:
        self.schema_detector = SchemaDetector()

    def _read_frame(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return pd.DataFrame(rows)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            return pd.DataFrame(rows)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported dataset extension: {suffix}")

    def load(self, file_path: str) -> list[dict]:
        path = Path(file_path)
        frame = self._read_frame(path)
        frame.columns = [str(c) for c in frame.columns]

        schema = self.schema_detector.detect(list(frame.columns))

        records: list[dict] = []
        for idx, row in frame.iterrows():
            sample_id = str(row.get(schema.id_field) or f"sample_{idx:06d}")
            input_text = normalize_text(str(row.get(schema.text_field, "")))
            if not input_text:
                continue
            instruction = ""
            if schema.task_field:
                instruction = normalize_text(str(row.get(schema.task_field, "")))
            options: list[str] = []
            if schema.label_options_field:
                raw_options = row.get(schema.label_options_field)
                if isinstance(raw_options, str):
                    options = [o.strip() for o in raw_options.split("|") if o.strip()]
                elif isinstance(raw_options, list):
                    options = [str(o).strip() for o in raw_options if str(o).strip()]

            records.append(
                {
                    "id": sample_id,
                    "input_text": input_text,
                    "task_instruction": instruction or "Classify the document using the provided label schema.",
                    "label_options": options,
                    "metadata": {
                        "source_file": path.name,
                        "row_index": int(idx),
                    },
                }
            )

        deduped = {item["id"]: item for item in records}
        return list(deduped.values())
