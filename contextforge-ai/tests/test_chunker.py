from app.core.chunker import Chunker


def test_chunker_outputs_overlap_chunks() -> None:
    text = " ".join([f"token{i}" for i in range(1200)])
    chunker = Chunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk("doc_001", text)

    assert len(chunks) > 2
    assert chunks[0]["chunk_id"].startswith("doc_001_chunk_")
    assert chunks[0]["importance_score"] > 0
