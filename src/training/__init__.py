from typing import Any, Callable
import random
from collections import defaultdict
from common.types import (
    AnnotationResult, DataRow, Example,
)
from common.logging import get_logger

logger = get_logger(__name__)


class ActiveLearningEngine:
    def __init__(self, uncertainty_threshold: float = 0.3):
        self.uncertainty_threshold = uncertainty_threshold

    def select_uncertain_samples(
        self,
        annotations: list[AnnotationResult],
        data: list[DataRow],
        num_samples: int = 10,
    ) -> list[DataRow]:
        uncertain_samples: list[tuple[DataRow, float]] = []

        for ann, row in zip(annotations, data):
            if not ann.labels:
                uncertain_samples.append((row, 1.0))
                continue

            confidence = ann.labels[0].confidence
            uncertainty = 1 - confidence

            if uncertainty >= self.uncertainty_threshold:
                uncertain_samples.append((row, uncertainty))

        uncertain_samples.sort(key=lambda x: x[1], reverse=True)
        return [row for row, _ in uncertain_samples[:num_samples]]

    def select_disagreement_samples(
        self,
        annotations: list[AnnotationResult],
        data: list[DataRow],
        num_samples: int = 10,
    ) -> list[DataRow]:
        label_counts: dict[str, int] = defaultdict(int)

        for ann in annotations:
            if ann.labels:
                label = str(ann.labels[0].value)
                label_counts[label] += 1

        disagreement_scores: list[tuple[DataRow, float]] = []

        for ann, row in zip(annotations, data):
            if not ann.labels or len(label_counts) < 2:
                continue

            label = str(ann.labels[0].value)
            label_freq = label_counts[label] / len(annotations)
            disagreement = 1 - label_freq

            if disagreement > 0.2:
                disagreement_scores.append((row, disagreement))

        disagreement_scores.sort(key=lambda x: x[1], reverse=True)
        return [row for row, _ in disagreement_scores[:num_samples]]

    def select_weak_label_samples(
        self,
        annotations: list[AnnotationResult],
        data: list[DataRow],
        num_samples: int = 10,
    ) -> list[DataRow]:
        weak_samples: list[tuple[DataRow, list[str]]] = []

        for ann, row in zip(annotations, data):
            if not ann.labels:
                weak_samples.append((row, ["no_label"]))
                continue

            label = ann.labels[0]
            issues = []

            if label.confidence < 0.5:
                issues.append("low_confidence")
            if not label.evidence_spans:
                issues.append("no_evidence")
            if not label.rationale:
                issues.append("no_rationale")
            if len(label.alternative_hypotheses) > 2:
                issues.append("too_many_alternatives")

            if issues:
                weak_samples.append((row, issues))

        weak_samples.sort(key=lambda x: len(x[1]), reverse=True)
        return [row for row, _ in weak_samples[:num_samples]]


class SyntheticExampleGenerator:
    def __init__(self, template: str = ""):
        self.template = template

    def generate(
        self,
        existing_examples: list[Example],
        num_new: int = 5,
    ) -> list[Example]:
        new_examples = []

        for _ in range(num_new):
            base_example = random.choice(existing_examples) if existing_examples else None

            if base_example:
                new_ex = Example(
                    input_text=self._perturb_text(base_example.input_text),
                    output=base_example.output,
                    rationale=base_example.rationale,
                    is_counterexample=random.random() > 0.7,
                )
            else:
                new_ex = Example(
                    input_text="Synthetic example text",
                    output="synthetic_label",
                    rationale="Auto-generated",
                    is_counterexample=False,
                )

            new_examples.append(new_ex)

        return new_examples

    def _perturb_text(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text

        num_perturbations = max(1, len(words) // 10)
        indices = random.sample(range(len(words)), min(num_perturbations, len(words)))

        perturbations = ["variation", "alternative", "modified", "changed"]
        for idx in indices:
            words[idx] = random.choice(perturbations)

        return " ".join(words)

    def generate_counterexamples(
        self,
        positive_examples: list[Example],
        num: int = 3,
    ) -> list[Example]:
        counterexamples = []

        for _ in range(num):
            base = random.choice(positive_examples) if positive_examples else None
            if base:
                counterexamples.append(Example(
                    input_text=self._invert_text(base.input_text),
                    output=f"not_{base.output}",
                    rationale="Counterexample generated by inversion",
                    is_counterexample=True,
                ))

        return counterexamples

    def _invert_text(self, text: str) -> str:
        replacements = {
            "positive": "negative",
            "good": "bad",
            "correct": "incorrect",
            "true": "false",
            "yes": "no",
        }

        result = text.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result


class PromptEvolutionEngine:
    def __init__(self):
        self.population: list[dict[str, Any]] = []
        self.generation = 0

    def initialize_population(
        self,
        base_prompt: str,
        population_size: int = 10,
    ) -> None:
        self.population = []

        variations = [
            base_prompt,
            base_prompt.replace("##", "#"),
            base_prompt + "\n\nBe concise.",
            base_prompt + "\n\nProvide detailed explanations.",
            base_prompt.replace("Your Response", "Answer the following:"),
            base_prompt.replace("rationale", "reasoning"),
            base_prompt.split("\n\n")[0] + "\n\n" + "\n\n".join(base_prompt.split("\n\n")[1:]),
        ]

        for i in range(population_size):
            variant = variations[i % len(variations)] if variations else base_prompt
            self.population.append({
                "prompt": variant,
                "fitness": 0.0,
                "generation": 0,
            })

    def evolve(
        self,
        fitness_fn: Callable,
        num_generations: int = 5,
        mutation_rate: float = 0.1,
    ) -> str:
        for gen in range(num_generations):
            for individual in self.population:
                fitness = fitness_fn(individual["prompt"])
                individual["fitness"] = fitness

            self.population.sort(key=lambda x: x["fitness"], reverse=True)

            top = self.population[: len(self.population) // 2]
            offspring = []

            while len(offspring) < len(self.population):
                parent1 = random.choice(top)
                parent2 = random.choice(top)
                child_prompt = self._crossover(parent1["prompt"], parent2["prompt"])

                if random.random() < mutation_rate:
                    child_prompt = self._mutate(child_prompt)

                offspring.append({
                    "prompt": child_prompt,
                    "fitness": 0.0,
                    "generation": gen + 1,
                })

            self.population = offspring
            self.generation = gen + 1

            logger.info("evolution_generation", generation=gen, best_fitness=self.population[0]["fitness"])

        best = max(self.population, key=lambda x: x["fitness"])
        return best["prompt"]

    def _crossover(self, p1: str, p2: str) -> str:
        lines1 = p1.split("\n\n")
        lines2 = p2.split("\n\n")

        if len(lines1) < 2:
            return p1

        split_idx = random.randint(1, len(lines1) - 1)
        return "\n\n".join(lines1[:split_idx] + lines2[split_idx:])

    def _mutate(self, prompt: str) -> str:
        mutations = [
            lambda p: p + "\n\nThink step by step.",
            lambda p: p.replace(".", "?"),
            lambda p: p.split("\n")[0] + "\n" + "\n".join(p.split("\n")[1:]),
            lambda p: "IMPORTANT: " + p,
        ]

        return random.choice(mutations)(prompt)


class GeneticPromptSearch:
    def __init__(self):
        self.engine = PromptEvolutionEngine()

    def search(
        self,
        base_prompt: str,
        examples: list[dict[str, Any]],
        metric_fn: Callable,
    ) -> str:
        self.engine.initialize_population(base_prompt)

        def fitness(prompt: str) -> float:
            return metric_fn(prompt, examples)

        best_prompt = self.engine.evolve(fitness)
        return best_prompt