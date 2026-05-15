"""Adaptive example selection with BM25/vector hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.core.pgvector_store import PGVectorStore
from app.core.vector_store import FaissIndexStore

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


@dataclass(slots=True)
class RetrievalExample:
    input_text: str
    label: str
    reason: str


class RetrievalEngine:
    def __init__(self) -> None:
        self.vectorizer = (
            TfidfVectorizer(max_features=6000, ngram_range=(1, 2)) if TfidfVectorizer else None
        )
        self.faiss_store = FaissIndexStore(str(settings.faiss_index_dir))
        self.pgvector_store = PGVectorStore(settings.database_url, settings.pgvector_table)
        self._cache_size = -1
        self.faiss_store.load()

    @staticmethod
    def _lexical_overlap_score(query: str, doc: str) -> float:
        q = set(query.lower().split())
        d = set(doc.lower().split())
        if not q or not d:
            return 0.0
        return len(q & d) / len(q)

    def select_examples(
        self,
        query_text: str,
        pool: list[dict],
        top_k: int = 6,
    ) -> dict:
        if not pool:
            return {"selected_examples": []}

        self._sync_indexes(pool)

        corpus = [p["input_text"] for p in pool]
        tokenized = [doc.lower().split() for doc in corpus]

        if BM25Okapi is not None:
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(query_text.lower().split())
        else:
            bm25_scores = np.array([self._lexical_overlap_score(query_text, doc) for doc in corpus])

        if self.vectorizer is not None:
            tfidf = self.vectorizer.fit_transform(corpus + [query_text])
            query_vec = tfidf[-1]
            doc_vecs = tfidf[:-1]
            cosine = (doc_vecs @ query_vec.T).toarray().ravel()
        else:
            cosine = np.array([self._lexical_overlap_score(query_text, doc) for doc in corpus])

        query_vector = self.faiss_store.embed_query(query_text)

        vector_rows = self._vector_candidates(query_text, query_vector, top_k=max(top_k * 2, 8))
        vector_bonus = {row["input"]: row.get("vector_score", 0.0) for row in vector_rows}

        bm25_n = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-9)
        cos_n = (cosine - np.min(cosine)) / (np.ptp(cosine) + 1e-9)
        hybrid = 0.45 * bm25_n + 0.35 * cos_n

        bonus = np.array([vector_bonus.get(text[:1200], 0.0) for text in corpus], dtype=np.float32)
        if np.ptp(bonus) > 0:
            bonus = (bonus - np.min(bonus)) / (np.ptp(bonus) + 1e-9)
        hybrid = hybrid + 0.2 * bonus

        ranked = np.argsort(hybrid)[::-1]
        selected: list[dict] = []
        labels: set[str] = set()
        for idx in ranked:
            item = pool[int(idx)]
            label = str(item.get("label", "unknown"))
            if len(selected) >= top_k:
                break
            if selected and label in labels and len(labels) > 1:
                continue
            labels.add(label)
            selected.append(
                {
                    "input": item["input_text"][:1200],
                    "label": label,
                    "reason": "high hybrid relevance and label diversity",
                }
            )

        return {"selected_examples": selected}

    def _sync_indexes(self, pool: list[dict]) -> None:
        if len(pool) == self._cache_size:
            return

        self.faiss_store.build_or_update(pool)
        self.faiss_store.persist()

        if settings.retrieval_backend.lower() == "pgvector" and self.pgvector_store.enabled:
            docs = self.faiss_store.docs
            vectors = self.faiss_store.vectors.tolist() if self.faiss_store.vectors is not None else []
            if docs and vectors:
                self.pgvector_store.upsert_many(docs, vectors)

        self._cache_size = len(pool)

    def _vector_candidates(self, query_text: str, query_vector: list[float], top_k: int) -> list[dict]:
        backend = settings.retrieval_backend.lower()
        if backend == "pgvector" and self.pgvector_store.enabled and query_vector:
            try:
                rows = self.pgvector_store.search(query_vector, top_k=top_k)
                if rows:
                    return rows
            except Exception:
                pass
        return self.faiss_store.search(query_text, top_k=top_k)
