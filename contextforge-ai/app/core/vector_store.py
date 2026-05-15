"""Persistent FAISS-backed vector store for retrieval examples."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


@dataclass(slots=True)
class VectorDoc:
    doc_id: str
    input_text: str
    label: str
    metadata: dict


class FaissIndexStore:
    """Stores vectors and metadata with disk persistence."""

    def __init__(self, index_dir: str = "data/processed/faiss") -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "metadata.json"
        self.vec_path = self.index_dir / "vectors.npy"
        self.vectorizer_path = self.index_dir / "vectorizer.pkl"

        self.docs: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2)) if TfidfVectorizer else None
        self.index = None

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def _embed_texts(self, texts: list[str], fit: bool) -> np.ndarray:
        if self.vectorizer is None:
            # fallback embedding if sklearn is unavailable
            features = np.array([[float(len(t)), float(len(t.split()))] for t in texts], dtype=np.float32)
            return self._normalize(features)
        if fit:
            mat = self.vectorizer.fit_transform(texts)
        else:
            mat = self.vectorizer.transform(texts)
        return self._normalize(mat.toarray().astype(np.float32))

    def build_or_update(self, pool: list[dict]) -> None:
        self.docs = []
        for idx, item in enumerate(pool):
            self.docs.append(
                {
                    "doc_id": str(item.get("id", f"ex_{idx:06d}")),
                    "input_text": str(item.get("input_text", "")),
                    "label": str(item.get("label", "unknown")),
                    "metadata": item.get("metadata", {}),
                }
            )

        texts = [d["input_text"] for d in self.docs]
        if not texts:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            self.index = None
            return

        self.vectors = self._embed_texts(texts, fit=True)
        dims = self.vectors.shape[1]

        if faiss is not None:
            self.index = faiss.IndexFlatIP(dims)
            self.index.add(self.vectors)
        else:
            self.index = None

    def persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.docs, ensure_ascii=True), encoding="utf-8")

        if self.vectorizer is not None:
            with self.vectorizer_path.open("wb") as f:
                pickle.dump(self.vectorizer, f)

        if self.vectors is None:
            np.save(self.vec_path, np.zeros((0, 0), dtype=np.float32))
        else:
            np.save(self.vec_path, self.vectors)

        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

    def load(self) -> bool:
        if not self.meta_path.exists() or not self.vec_path.exists():
            return False

        self.docs = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vectors = np.load(self.vec_path)

        if self.vectorizer_path.exists() and self.vectorizer is not None:
            with self.vectorizer_path.open("rb") as f:
                self.vectorizer = pickle.load(f)

        if faiss is not None and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        elif self.vectors.size > 0:
            self.index = None
        return True

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        if not self.docs:
            return []

        query_vec = self._embed_texts([query], fit=False)[0]
        query_vec_2d = np.expand_dims(query_vec, axis=0)

        if faiss is not None and self.index is not None:
            scores, idxs = self.index.search(query_vec_2d, min(top_k, len(self.docs)))
            selected = []
            for score, idx in zip(scores[0], idxs[0], strict=False):
                if idx < 0:
                    continue
                doc = self.docs[int(idx)]
                selected.append({
                    "input": doc["input_text"][:1200],
                    "label": doc["label"],
                    "doc_id": doc["doc_id"],
                    "vector_score": float(score),
                })
            return selected

        if self.vectors is None or self.vectors.size == 0:
            return []

        sims = self.vectors @ query_vec
        order = np.argsort(sims)[::-1][:top_k]
        rows: list[dict] = []
        for idx in order:
            doc = self.docs[int(idx)]
            rows.append(
                {
                    "input": doc["input_text"][:1200],
                    "label": doc["label"],
                    "doc_id": doc["doc_id"],
                    "vector_score": float(sims[int(idx)]),
                }
            )
        return rows

    def embed_query(self, query: str) -> list[float]:
        if not self.docs:
            return []
        vector = self._embed_texts([query], fit=False)[0]
        return vector.astype(np.float32).tolist()
