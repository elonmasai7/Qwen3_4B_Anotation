"""Prompt rendering for structured annotation flows."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class PromptBuilder:
    def __init__(self, templates_dir: str = "app/prompts") -> None:
        provided = Path(templates_dir)
        if provided.exists():
            self.templates_dir = provided
        else:
            self.templates_dir = Path(__file__).resolve().parents[1] / "prompts"
        self.env = Environment(loader=FileSystemLoader(self.templates_dir), autoescape=False)

    def build_annotation_prompt(
        self,
        task_instruction: str,
        label_schema: list[str],
        examples: list[dict],
        memory: dict,
        evidence_chunks: list[dict],
    ) -> str:
        template = self.env.get_template("base_annotation_prompt.jinja2")
        return template.render(
            task_instruction=task_instruction,
            label_schema=label_schema,
            examples=examples,
            memory=memory,
            evidence_chunks=evidence_chunks,
        )
