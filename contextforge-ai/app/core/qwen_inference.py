"""Qwen3-4B inference wrapper with safe local fallback."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import settings


class QwenInference:
    def __init__(self) -> None:
        self.model_name = settings.qwen_model_name
        self.use_vllm = settings.use_vllm
        self._generator = None
        self._backend = "mock"
        self._init_backend()

    def _init_backend(self) -> None:
        if self.use_vllm:
            try:
                from vllm import LLM

                self._generator = LLM(model=self.model_name)
                self._backend = "vllm"
                return
            except Exception:
                self._backend = "mock"

        try:
            from transformers import pipeline

            self._generator = pipeline("text-generation", model=self.model_name)
            self._backend = "transformers"
        except Exception:
            self._backend = "mock"

    def _mock_response(self, prompt: str) -> dict[str, Any]:
        # Deterministic local fallback to keep system runnable without model download.
        label = "unknown"
        if "positive" in prompt.lower() and "negative" in prompt.lower():
            label = "positive"
        return {
            "label": label,
            "confidence": 0.62,
            "evidence": [prompt[:180]],
            "rationale": "Mock fallback selected due to unavailable model backend.",
            "uncertainty_reason": "local_fallback_backend",
        }

    def generate_json(self, prompt: str, retries: int = 2, timeout_s: float = 120.0) -> dict:
        for _ in range(retries + 1):
            start = time.time()
            try:
                if self._backend == "mock":
                    return self._mock_response(prompt)

                if self._backend == "transformers":
                    output = self._generator(
                        prompt,
                        max_new_tokens=settings.max_new_tokens,
                        do_sample=True,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                    )[0]["generated_text"]
                else:
                    from vllm import SamplingParams

                    params = SamplingParams(
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        max_tokens=settings.max_new_tokens,
                    )
                    output = self._generator.generate([prompt], params)[0].outputs[0].text

                raw = output[output.find("{") : output.rfind("}") + 1]
                parsed = json.loads(raw)
                if "label" not in parsed:
                    raise ValueError("Missing label")
                return parsed
            except Exception:
                if time.time() - start > timeout_s:
                    break
                continue

        return self._mock_response(prompt)
