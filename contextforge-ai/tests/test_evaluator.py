from app.core.evaluator import Evaluator


def test_evaluator_metrics() -> None:
    pred = [
        {"id": "1", "label": "A", "confidence": 0.9},
        {"id": "2", "label": "B", "confidence": 0.4},
    ]
    gold = [
        {"id": "1", "label": "A"},
        {"id": "2", "label": "A"},
    ]
    report = Evaluator().evaluate(pred, gold, latency_s=1.0)
    assert 0.0 <= report["accuracy"] <= 1.0
    assert "confusion_matrix" in report
