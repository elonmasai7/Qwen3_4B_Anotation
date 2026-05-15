from pathlib import Path

from app.core.vector_store import FaissIndexStore


def test_faiss_store_persistence_and_search(tmp_path: Path) -> None:
    store = FaissIndexStore(index_dir=str(tmp_path / "faiss_index"))
    pool = [
        {"id": "1", "input_text": "great product and excellent quality", "label": "positive"},
        {"id": "2", "input_text": "bad service and late delivery", "label": "negative"},
    ]
    store.build_or_update(pool)
    store.persist()

    loaded = FaissIndexStore(index_dir=str(tmp_path / "faiss_index"))
    assert loaded.load()
    rows = loaded.search("excellent product", top_k=1)
    assert len(rows) == 1
    assert "label" in rows[0]
