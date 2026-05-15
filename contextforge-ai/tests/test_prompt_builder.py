from pathlib import Path

from app.core.prompt_builder import PromptBuilder


def test_prompt_builder_renders_required_sections() -> None:
    builder = PromptBuilder(templates_dir=str(Path("app/prompts")))
    prompt = builder.build_annotation_prompt(
        task_instruction="Choose a label",
        label_schema=["yes", "no"],
        examples=[{"input": "a", "label": "yes", "reason": "x"}],
        memory={"global_summary": "sum"},
        evidence_chunks=[{"text": "evidence"}],
    )
    assert "Task:" in prompt
    assert "Label Schema:" in prompt
    assert "Output valid JSON only" in prompt
