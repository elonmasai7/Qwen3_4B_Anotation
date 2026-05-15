"""Primary annotation agent."""

from __future__ import annotations

from app.core.qwen_inference import QwenInference


class AnnotationAgent:
    def __init__(self, inference: QwenInference | None = None) -> None:
        self.inference = inference or QwenInference()

    def annotate(self, prompt: str) -> dict:
        return self.inference.generate_json(prompt)
