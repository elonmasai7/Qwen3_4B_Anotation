from app.core.retrieval_engine import RetrievalEngine


def test_hybrid_retrieval_returns_selected_examples() -> None:
    engine = RetrievalEngine()
    pool = [
        {"input_text": "The movie was wonderful and uplifting", "label": "positive"},
        {"input_text": "The story was dull and slow", "label": "negative"},
        {"input_text": "A neutral summary of events", "label": "neutral"},
    ]
    result = engine.select_examples("uplifting story", pool, top_k=2)
    assert "selected_examples" in result
    assert len(result["selected_examples"]) >= 1
