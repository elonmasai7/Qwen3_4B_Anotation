"""Prediction and submission export utilities."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


class Exporter:
    def export_predictions_jsonl(self, rows: list[dict], path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def build_submission_bundle(
        self,
        predictions: list[dict],
        config: dict,
        technical_report_path: str,
        source_root: str,
        output_dir: str = "submission",
    ) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.export_predictions_jsonl(predictions, str(out / "predictions.jsonl"))
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        shutil.copyfile(technical_report_path, out / "technical_report.md")

        zip_path = out / "source_code.zip"
        source_root_path = Path(source_root).resolve()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in source_root_path.rglob("*"):
                if path.is_dir():
                    continue
                rel = path.relative_to(source_root_path)
                rel_str = str(rel)
                if rel_str.startswith("submission/"):
                    continue
                if any(part in {".git", "__pycache__", ".pytest_cache", ".venv"} for part in rel.parts):
                    continue
                zf.write(path, rel)
        return str(out)
