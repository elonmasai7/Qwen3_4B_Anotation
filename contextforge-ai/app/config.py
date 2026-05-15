"""Runtime configuration for ContextForge AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    app_name: str = "ContextForge AI"
    app_env: str = os.getenv("CONTEXTFORGE_ENV", "local")
    data_dir: Path = Path(os.getenv("CONTEXTFORGE_DATA_DIR", "data"))

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./contextforge.db")
    use_postgres: bool = _as_bool(os.getenv("USE_POSTGRES"), False)

    qwen_model_name: str = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-4B")
    use_vllm: bool = _as_bool(os.getenv("USE_VLLM"), False)
    device: str = os.getenv("DEVICE", "auto")

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    retrieval_backend: str = os.getenv("RETRIEVAL_BACKEND", "faiss")
    faiss_index_dir: Path = Path(os.getenv("FAISS_INDEX_DIR", "data/processed/faiss"))
    pgvector_table: str = os.getenv("PGVECTOR_TABLE", "example_embeddings")
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "2048"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "256"))

    queue_backend: str = os.getenv("QUEUE_BACKEND", "sqlite")
    queue_poll_interval_s: float = float(os.getenv("QUEUE_POLL_INTERVAL_S", "0.2"))

    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))


settings = Settings()
