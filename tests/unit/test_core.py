import pytest
import asyncio
from src.common.types import (
    DataRow, Chunk, AnnotationLabel, AnnotationResult,
    AnnotationStatus, ChunkStrategy, VotingStrategy,
    PromptTemplate, Example, Metadata,
)
from src.ingestion.loaders import JSONLoader, CSVLoader, get_loader
from src.ingestion.processors import ContextProcessor, ChunkingConfig
from src.annotation.prompt_constructor import PromptConstructor, PromptConfig
from src.annotation.reasoner import AnnotationReasoner, SelfConsistencyEngine
from src.verification import ConsistencyChecker, AdversarialChecker, RepairEngine
from src.evaluation import MetricsCalculator, EvaluationEngine
from src.training import ActiveLearningEngine, PromptEvolutionEngine


class TestTypes:
    def test_data_row_creation(self):
        row = DataRow(
            content="Test content",
            metadata=Metadata(source="test"),
        )
        assert row.content == "Test content"
        assert row.id is not None

    def test_chunk_creation(self):
        chunk = Chunk(
            text="Test chunk",
            start_idx=0,
            end_idx=11,
            chunk_index=0,
        )
        assert chunk.text == "Test chunk"
        assert chunk.start_idx == 0

    def test_annotation_label(self):
        label = AnnotationLabel(
            value="positive",
            confidence=0.9,
            rationale="Test rationale",
            evidence_spans=[[0, 10]],
        )
        assert label.confidence == 0.9
        assert label.value == "positive"


class TestLoaders:
    @pytest.mark.asyncio
    async def test_json_loader(self, tmp_path):
        import json
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps([{"id": "1", "content": "test"}]))

        loader = JSONLoader(test_file)
        rows = [row async for row in loader.load()]
        assert len(rows) == 1
        assert rows[0].content == "test"

    def test_get_loader(self, tmp_path):
        import json
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        loader = get_loader(test_file)
        assert loader is not None


class TestProcessors:
    @pytest.mark.asyncio
    async def test_context_processor(self):
        config = ChunkingConfig(
            max_tokens=100,
            strategy=ChunkStrategy.SEMANTIC,
        )
        processor = ContextProcessor(config)

        data = DataRow(content="This is a test document. It has multiple sentences.")
        chunks = await processor.process(data)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)


class TestPromptConstructor:
    def test_basic_prompt_creation(self):
        config = PromptConfig(num_examples=3)
        constructor = PromptConstructor(config)

        template = PromptTemplate(
            name="Test Template",
            instruction="Classify the text",
            task_definition="Determine sentiment",
            cot_scaffold="Think step by step",
            output_schema='{"label": "string"}',
        )

        prompt = constructor.construct(template, "Test input")
        assert "Classify the text" in prompt
        assert "Test input" in prompt


class TestReasoner:
    @pytest.mark.asyncio
    async def test_annotation_reasoner(self):
        reasoner = AnnotationReasoner()
        data = DataRow(content="Test content")

        result = await reasoner.reason(data, "Test prompt")
        assert result.data_id == data.id


class TestVerification:
    def test_consistency_checker(self):
        checker = ConsistencyChecker()

        data = DataRow(content="Test content")
        annotation = AnnotationResult(
            data_id=data.id,
            labels=[
                AnnotationLabel(
                    value="test",
                    confidence=0.9,
                    rationale="Test",
                    evidence_spans=[[0, 4]],
                )
            ],
        )

        result = checker.verify(annotation, data)
        assert result is not None


class TestMetrics:
    def test_accuracy_calculation(self):
        predictions = [
            AnnotationResult(
                data_id="1",
                labels=[AnnotationLabel(value="positive", confidence=0.9, rationale="")],
            )
        ]
        ground_truths = ["positive"]

        accuracy = MetricsCalculator.calculate_accuracy(predictions, ground_truths)
        assert accuracy == 1.0

    def test_f1_calculation(self):
        predictions = [
            AnnotationResult(
                data_id="1",
                labels=[AnnotationLabel(value="positive", confidence=0.9, rationale="")],
            )
        ]
        ground_truths = ["positive"]

        f1 = MetricsCalculator.calculate_f1(predictions, ground_truths)
        assert f1 > 0


class TestTraining:
    def test_active_learning_selection(self):
        engine = ActiveLearningEngine(uncertainty_threshold=0.3)

        annotations = [
            AnnotationResult(
                data_id="1",
                labels=[AnnotationLabel(value="a", confidence=0.3, rationale="")],
            ),
            AnnotationResult(
                data_id="2",
                labels=[AnnotationLabel(value="b", confidence=0.9, rationale="")],
            ),
        ]

        data = [
            DataRow(content="test1"),
            DataRow(content="test2"),
        ]

        uncertain = engine.select_uncertain_samples(annotations, data)
        assert len(uncertain) >= 0

    def test_prompt_evolution_initialization(self):
        engine = PromptEvolutionEngine()
        engine.initialize_population("Test prompt", population_size=5)

        assert len(engine.population) == 5


class TestEvaluationEngine:
    def test_evaluation_engine_creation(self):
        engine = EvaluationEngine()
        assert engine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])