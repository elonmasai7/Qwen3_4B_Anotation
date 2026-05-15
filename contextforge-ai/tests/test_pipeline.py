from app.core.annotation_pipeline import AnnotationPipeline


def test_annotation_pipeline_end_to_end() -> None:
    pipeline = AnnotationPipeline()
    sample = {
        "id": "sample_x",
        "input_text": "The document states the outcome is excellent and beneficial.",
        "task_instruction": "Classify sentiment",
        "label_options": ["positive", "negative", "neutral"],
    }
    pred, latency = pipeline.annotate_sample(sample, example_pool=[])

    assert pred["id"] == "sample_x"
    assert pred["label"] in sample["label_options"]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert latency >= 0.0
