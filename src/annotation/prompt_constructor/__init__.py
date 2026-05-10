from typing import Any
from dataclasses import dataclass
from common.types import Example, PromptTemplate, Chunk
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class PromptConfig:
    include_cot: bool = True
    include_schema: bool = True
    num_examples: int = 5
    dynamic_examples: bool = True


class PromptConstructor:
    def __init__(self, config: PromptConfig | None = None):
        self.config = config or PromptConfig()

    def construct(
        self,
        template: PromptTemplate,
        input_text: str,
        retrieved_chunks: list[Chunk] | None = None,
        examples: list[Example] | None = None,
    ) -> str:
        parts = []

        parts.append(self._format_instruction(template.instruction))
        parts.append(self._format_task_definition(template.task_definition))

        if self.config.include_cot:
            parts.append(self._format_cot_scaffold(template.cot_scaffold))

        if examples or retrieved_chunks:
            parts.append(self._format_examples(
                examples or [],
                retrieved_chunks,
                template.examples,
            ))

        if self.config.include_schema:
            parts.append(self._format_output_schema(template.output_schema))

        parts.append(self._format_input(input_text))

        return "\n\n".join(parts)

    def _format_instruction(self, instruction: str) -> str:
        return f"""## Instruction
{instruction}"""

    def _format_task_definition(self, task_def: str) -> str:
        return f"""## Task Definition
{task_def}"""

    def _format_cot_scaffold(self, scaffold: str) -> str:
        return f"""## Chain-of-Thought Reasoning
{scaffold}"""

    def _format_examples(
        self,
        dynamic_examples: list[Example],
        chunks: list[Chunk] | None,
        template_examples: list[Example],
    ) -> str:
        all_examples = dynamic_examples[: self.config.num_examples]

        if not all_examples and template_examples:
            all_examples = template_examples[: self.config.num_examples]

        if not all_examples and chunks:
            all_examples = self._create_examples_from_chunks(chunks)

        if not all_examples:
            return "## Examples\n(No examples available)"

        example_parts = ["## Examples\n"]
        for i, ex in enumerate(all_examples):
            prefix = "Counter-example" if ex.is_counterexample else "Example"
            example_parts.append(f"\n### {prefix} {i + 1}")
            example_parts.append(f"**Input:** {ex.input_text[:300]}...")
            example_parts.append(f"**Output:** {ex.output}")
            if ex.rationale:
                example_parts.append(f"**Rationale:** {ex.rationale}")

        return "\n".join(example_parts)

    def _create_examples_from_chunks(self, chunks: list[Chunk]) -> list[Example]:
        examples = []
        for chunk in chunks[: self.config.num_examples]:
            if chunk.importance_score and chunk.importance_score > 0.5:
                examples.append(Example(
                    input_text=chunk.text[:200],
                    output="",
                    is_counterexample=False,
                ))
        return examples

    def _format_output_schema(self, schema: str) -> str:
        return f"""## Output Schema
```json
{schema}
```"""

    def _format_input(self, input_text: str) -> str:
        return f"""## Input Document
```
{input_text[:8000]}
```

## Your Response
Provide your annotation following the schema above. Include:
1. The label(s) with confidence scores
2. A brief rationale
3. Evidence spans from the document
4. Any alternative hypotheses considered"""


class DynamicExampleRetriever:
    def __init__(self, embedding_model_name: str | None = None):
        self.embedding_model_name = embedding_model_name or settings.retrieval.embedding_model

    def retrieve_similar(
        self,
        query_text: str,
        examples: list[Example],
        top_k: int | None = None,
    ) -> list[Example]:
        if not examples or not any(e.embedding for e in examples):
            return examples[: top_k or settings.retrieval.top_k]

        query_embedding = self._get_embedding(query_text)
        if query_embedding is None:
            return examples[: top_k or settings.retrieval.top_k]

        similarities = []
        for ex in examples:
            if ex.embedding:
                sim = self._cosine_similarity(query_embedding, ex.embedding)
                similarities.append((ex, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in similarities[: top_k or settings.retrieval.top_k]]

    def retrieve_bm25(
        self,
        query_text: str,
        examples: list[Example],
        top_k: int | None = None,
    ) -> list[Example]:
        query_terms = query_text.lower().split()
        scores = []

        for ex in examples:
            score = sum(1 for t in query_terms if t in ex.input_text.lower())
            scores.append((ex, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scores[: top_k or settings.retrieval.top_k]]

    def retrieve_hybrid(
        self,
        query_text: str,
        examples: list[Example],
        top_k: int | None = None,
    ) -> list[Example]:
        semantic_results = self.retrieve_similar(query_text, examples, top_k * 2)
        bm25_results = self.retrieve_bm25(query_text, examples, top_k * 2)

        combined_scores: dict[str, float] = {}
        for ex in semantic_results:
            combined_scores[ex.id] = combined_scores.get(ex.id, 0) + 0.5

        for ex in bm25_results:
            combined_scores[ex.id] = combined_scores.get(ex.id, 0) + 0.5

        sorted_examples = sorted(
            [(ex, combined_scores[ex.id]) for ex in examples if ex.id in combined_scores],
            key=lambda x: x[1],
            reverse=True,
        )

        return [ex for ex, _ in sorted_examples[: top_k or settings.retrieval.top_k]]

    def _get_embedding(self, text: str) -> list[float] | None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
            return model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.warning("embedding_failed", error=str(e))
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))


class PromptOptimizer:
    def __init__(self):
        self.best_prompt: str | None = None
        self.best_score: float = 0.0

    def optimize(
        self,
        base_prompt: str,
        examples: list[dict[str, Any]],
        metric_fn: callable,
    ) -> str:
        variations = self._generate_variations(base_prompt)

        for variant in variations:
            score = metric_fn(variant, examples)
            if score > self.best_score:
                self.best_score = score
                self.best_prompt = variant

        return self.best_prompt or base_prompt

    def _generate_variations(self, prompt: str) -> list[str]:
        return [
            prompt,
            prompt.replace("##", "#"),
            prompt + "\n\nBe concise and accurate.",
            prompt.replace("Your Response", "Provide a detailed, structured response"),
        ]