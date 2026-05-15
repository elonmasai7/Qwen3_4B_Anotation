"""Optional pgvector retrieval backend for large corpora."""

from __future__ import annotations

import json

from sqlalchemy import create_engine, text


class PGVectorStore:
    def __init__(self, database_url: str, table_name: str = "example_embeddings") -> None:
        self.database_url = database_url
        self.table_name = table_name
        self.engine = create_engine(database_url, future=True)

    @property
    def enabled(self) -> bool:
        return self.database_url.startswith("postgresql")

    def bootstrap(self, dims: int) -> None:
        if not self.enabled:
            return
        safe_dims = int(dims)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        doc_id TEXT PRIMARY KEY,
                        input_text TEXT NOT NULL,
                        label TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({safe_dims}) NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding "
                    f"ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)"
                )
            )

    def upsert_many(self, docs: list[dict], vectors: list[list[float]]) -> None:
        if not self.enabled or not docs:
            return
        dims = len(vectors[0])
        self.bootstrap(dims)
        with self.engine.begin() as conn:
            for doc, vector in zip(docs, vectors, strict=False):
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self.table_name} (doc_id, input_text, label, metadata, embedding)
                        VALUES (:doc_id, :input_text, :label, CAST(:metadata AS JSONB), CAST(:embedding AS vector))
                        ON CONFLICT (doc_id)
                        DO UPDATE SET
                          input_text = EXCLUDED.input_text,
                          label = EXCLUDED.label,
                          metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "doc_id": doc["doc_id"],
                        "input_text": doc["input_text"],
                        "label": doc["label"],
                        "metadata": json.dumps(doc.get("metadata", {}), ensure_ascii=True),
                        "embedding": self._vector_literal(vector),
                    },
                )

    def search(self, query_vector: list[float], top_k: int = 6) -> list[dict]:
        if not self.enabled:
            return []
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT
                        doc_id,
                        input_text,
                        label,
                        metadata,
                        1 - (embedding <=> CAST(:embedding AS vector)) AS vector_score
                    FROM {self.table_name}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                {"embedding": self._vector_literal(query_vector), "top_k": top_k},
            )
            rows = result.fetchall()

        return [
            {
                "doc_id": row.doc_id,
                "input": row.input_text[:1200],
                "label": row.label,
                "metadata": row.metadata or {},
                "vector_score": float(row.vector_score),
            }
            for row in rows
        ]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"
