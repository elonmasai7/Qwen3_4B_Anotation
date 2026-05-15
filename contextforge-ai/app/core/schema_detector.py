"""Automatic schema detection for heterogeneous datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectedSchema:
    id_field: str
    text_field: str
    task_field: str | None
    label_options_field: str | None


class SchemaDetector:
    ID_CANDIDATES = ("id", "sample_id", "uid", "record_id")
    TEXT_CANDIDATES = ("input_text", "text", "document", "content", "input")
    TASK_CANDIDATES = ("task_instruction", "instruction", "task", "prompt")
    LABEL_CANDIDATES = ("label_options", "options", "labels", "classes")

    @staticmethod
    def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
        lowered = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
        for col in columns:
            lc = col.lower()
            if any(c in lc for c in candidates):
                return col
        return None

    def detect(self, columns: list[str]) -> DetectedSchema:
        id_field = self._pick(columns, self.ID_CANDIDATES) or columns[0]
        text_field = self._pick(columns, self.TEXT_CANDIDATES) or columns[min(len(columns) - 1, 1)]
        task_field = self._pick(columns, self.TASK_CANDIDATES)
        label_field = self._pick(columns, self.LABEL_CANDIDATES)
        return DetectedSchema(
            id_field=id_field,
            text_field=text_field,
            task_field=task_field,
            label_options_field=label_field,
        )
